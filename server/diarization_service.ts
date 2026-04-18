import { GoogleGenAI } from "@google/genai";
import { SettingsManager } from "./services/settings_manager.js";
import { SpeakerEmbeddingService } from "./speaker_embedding_service.js";
import { ClusteringService } from "./clustering_service.js";
import { PyannoteDiarizationService } from "./pyannote_diarization_service.js";
import { VadService } from "./vad_service.js";
import { spawn } from "child_process";
import fs from "fs-extra";
import path from "path";
import { PathManager } from "./path_manager.js";
import { resolveToolCommand } from "./runtime_tools.js";

export interface TranscribedChunk {
  start_ts: number;
  end_ts?: number;
  text: string;
  speaker?: string;
}

export interface DiarizationContext {
  speechSegments?: Array<{ start: number; end: number }>;
  vadWindows?: Array<{ start: number; end: number }>;
  options?: DiarizationOptions;
  onProgress?: (message: string) => void;
  signal?: AbortSignal;
}

export interface DiarizationOptions {
  provider?: 'classic' | 'pyannote';
  mode?: 'auto' | 'fixed' | 'range' | 'many';
  exactSpeakerCount?: number;
  minSpeakers?: number;
  maxSpeakers?: number;
  scenePreset?: 'interview' | 'podcast' | 'meeting' | 'presentation_qa' | 'custom';
  preferStablePrimarySpeaker?: boolean;
  allowShortInterjectionSpeaker?: boolean;
  preferVadBoundedRegions?: boolean;
  forceMergeTinyClustersInTwoSpeakerMode?: boolean;
  semanticFallbackEnabled?: boolean;
}

interface NormalizedDiarizationOptions {
  provider: 'classic' | 'pyannote';
  mode: 'auto' | 'fixed' | 'range' | 'many';
  exactSpeakerCount: number | null;
  minSpeakers: number;
  maxSpeakers: number;
  scenePreset: 'interview' | 'podcast' | 'meeting' | 'presentation_qa' | 'custom';
  preferStablePrimarySpeaker: boolean;
  allowShortInterjectionSpeaker: boolean;
  preferVadBoundedRegions: boolean;
  forceMergeTinyClustersInTwoSpeakerMode: boolean;
  semanticFallbackEnabled: boolean;
}

interface EmbeddingQuality {
  pairCount: number;
  minSimilarity: number;
  maxSimilarity: number;
  avgSimilarity: number;
  p90Similarity: number;
}

interface ChunkBounds {
  index: number;
  start: number;
  end: number;
}

interface AcousticRegion {
  start: number;
  end: number;
  chunkIndices: number[];
}

interface AcousticPassResult {
  assignedSpeakerIds: number[];
  uniqueSpeakerCount: number;
  quality: EmbeddingQuality;
  threshold: number;
  rawSpeakerIds: number[];
  postMergeSpeakerIds: number[];
  resegmentedChunkCount: number;
  resegmentationPasses: number;
}

class DiarizationLowConfidenceError extends Error {
  attemptedPasses: DiarizationDiagnostics["attemptedPasses"];
  selectedPass?: DiarizationDiagnostics["selectedPass"];

  constructor(
    message: string,
    details: {
      attemptedPasses: DiarizationDiagnostics["attemptedPasses"];
      selectedPass?: DiarizationDiagnostics["selectedPass"];
    }
  ) {
    super(message);
    this.name = "DiarizationLowConfidenceError";
    this.attemptedPasses = details.attemptedPasses;
    this.selectedPass = details.selectedPass;
  }
}

export interface DiarizationDiagnostics {
  provider: "acoustic" | "semantic";
  selectedSource: "speech_region" | "vad_chunk" | "chunk" | "pyannote" | "semantic";
  speechSegmentCount: number;
  vadWindowCount: number;
  options: NormalizedDiarizationOptions;
  selectedPass?: {
    source: string;
    regionCount: number;
    uniqueSpeakerCount: number;
    threshold: number;
    quality: EmbeddingQuality;
    resegmentedChunkCount?: number;
    resegmentationPasses?: number;
  };
  attemptedPasses: Array<{
    source: string;
    regionCount: number;
    uniqueSpeakerCount: number;
    threshold: number;
    quality: EmbeddingQuality;
    resegmentedChunkCount?: number;
    resegmentationPasses?: number;
  }>;
}

export interface DiarizationResult {
  chunks: TranscribedChunk[];
  diagnostics: DiarizationDiagnostics;
}

export class DiarizationService {
  private static readonly minClipDurationSec = 1.5;
  private static readonly maxClipDurationSec = 8;
  private static readonly targetRegionDurationSec = 4.5;
  private static readonly maxAssignedSpeakerCount = 6;
  private static readonly acousticClusterThresholdMin = 0.45;
  private static readonly acousticClusterThresholdMax = 0.6;
  private static readonly acousticLowConfidenceP90 = 0.94;
  private static readonly acousticLowConfidenceAvg = 0.9;
  private static readonly regionMergeGapSec = 0.2;
  private static readonly regionAssignOverlapMinSec = 0.1;
  private static readonly regionAssignPaddingSec = 0.2;
  private static vadSpeechSegmentsPromiseByAudioPath = new Map<string, Promise<Array<{ start: number; end: number }>>>();

  private static throwIfAborted(signal?: AbortSignal) {
    if (!signal?.aborted) return;
    throw new Error('Diarization aborted.');
  }

  private static getFfmpegBinary() {
    return resolveToolCommand("ffmpeg");
  }

  private static async getSpeechSegmentsForAudio(audioPath: string) {
    const cached = this.vadSpeechSegmentsPromiseByAudioPath.get(audioPath);
    if (cached) return cached;

    const pending = VadService.detectSpeech(audioPath).catch((error) => {
      this.vadSpeechSegmentsPromiseByAudioPath.delete(audioPath);
      throw error;
    });
    this.vadSpeechSegmentsPromiseByAudioPath.set(audioPath, pending);
    return pending;
  }

  static async performDiarization(
    chunks: TranscribedChunk[],
    audioPath: string,
    context: DiarizationContext = {}
  ): Promise<DiarizationResult> {
    this.throwIfAborted(context.signal);
    if (chunks.length === 0) {
      const normalizedOptions = this.normalizeOptions(context.options, chunks.length);
      return {
        chunks,
        diagnostics: {
          provider: "acoustic",
          selectedSource: "chunk",
          speechSegmentCount: context.speechSegments?.length ?? 0,
          vadWindowCount: context.vadWindows?.length ?? 0,
          options: normalizedOptions,
          attemptedPasses: [],
        },
      };
    }

    try {
      context.onProgress?.(`Speaker diarization started (${chunks.length} transcript chunks)...`);
      const normalizedOptions = this.normalizeOptions(context.options, chunks.length);
      if (normalizedOptions.provider === 'pyannote') {
        return await this.performPyannoteDiarization(chunks, audioPath, context, normalizedOptions);
      }
      return await this.performAcousticDiarization(chunks, audioPath, context);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.warn(`[Diarization] Acoustic diarization failed, falling back to semantic diarization: ${message}`);
      context.onProgress?.(`Acoustic diarization fallback triggered: ${message}`);
      const semantic = await this.performSemanticDiarization(chunks);
      const attemptedPasses = error instanceof DiarizationLowConfidenceError ? error.attemptedPasses : [];
      const selectedPass = error instanceof DiarizationLowConfidenceError ? error.selectedPass : undefined;
      return {
        chunks: semantic,
        diagnostics: {
          provider: "semantic",
          selectedSource: "semantic",
          speechSegmentCount: context.speechSegments?.length ?? 0,
          vadWindowCount: context.vadWindows?.length ?? 0,
          options: this.normalizeOptions(context.options, chunks.length),
          selectedPass,
          attemptedPasses,
        },
      };
    }
  }

  private static async performPyannoteDiarization(
    chunks: TranscribedChunk[],
    audioPath: string,
    context: DiarizationContext,
    normalizedOptions: NormalizedDiarizationOptions
  ): Promise<DiarizationResult> {
    this.throwIfAborted(context.signal);
    console.log(`[Diarization] Starting pyannote diarization: ${audioPath}`);
    context.onProgress?.('Preparing pyannote diarization...');

    const chunkBounds = this.normalizeChunkBounds(chunks);
    const pyannote = await PyannoteDiarizationService.diarizeAudio(
      audioPath,
      {
        exactSpeakerCount: normalizedOptions.exactSpeakerCount,
        minSpeakers: normalizedOptions.minSpeakers,
        maxSpeakers: normalizedOptions.maxSpeakers,
      },
      context.onProgress,
      context.signal
    );
    const assignedSpeakers = this.assignTurnsToChunkSpeakers(chunkBounds, pyannote.turns);
    const labelToSpeaker = new Map<string, number>();
    let nextSpeakerId = 0;
    const numericSpeakerIds = assignedSpeakers.map((label) => {
      const normalizedLabel = String(label || '').trim() || `Speaker ${nextSpeakerId + 1}`;
      if (!labelToSpeaker.has(normalizedLabel)) {
        labelToSpeaker.set(normalizedLabel, nextSpeakerId);
        nextSpeakerId += 1;
      }
      return labelToSpeaker.get(normalizedLabel)!;
    });

    return {
      chunks: chunks.map((chunk, index) => ({
        ...chunk,
        speaker: `Speaker ${numericSpeakerIds[index] + 1}`,
      })),
      diagnostics: {
        provider: "acoustic",
        selectedSource: "pyannote",
        speechSegmentCount: context.speechSegments?.length ?? 0,
        vadWindowCount: context.vadWindows?.length ?? 0,
        options: normalizedOptions,
        selectedPass: {
          source: "pyannote",
          regionCount: pyannote.turns.length,
          uniqueSpeakerCount: new Set(numericSpeakerIds).size,
          threshold: pyannote.diagnostics.clusterThreshold,
          quality: {
            pairCount: pyannote.diagnostics.similarity.pairCount,
            minSimilarity: pyannote.diagnostics.similarity.min,
            maxSimilarity: pyannote.diagnostics.similarity.max,
            avgSimilarity: pyannote.diagnostics.similarity.avg,
            p90Similarity: pyannote.diagnostics.similarity.p90,
          },
        },
        attemptedPasses: [
          {
            source: "pyannote",
            regionCount: pyannote.turns.length,
            uniqueSpeakerCount: new Set(numericSpeakerIds).size,
            threshold: pyannote.diagnostics.clusterThreshold,
            quality: {
              pairCount: pyannote.diagnostics.similarity.pairCount,
              minSimilarity: pyannote.diagnostics.similarity.min,
              maxSimilarity: pyannote.diagnostics.similarity.max,
              avgSimilarity: pyannote.diagnostics.similarity.avg,
              p90Similarity: pyannote.diagnostics.similarity.p90,
            },
          },
        ],
      },
    };
  }

  private static async performAcousticDiarization(
    chunks: TranscribedChunk[],
    audioPath: string,
    context: DiarizationContext
  ): Promise<DiarizationResult> {
    this.throwIfAborted(context.signal);
    console.log(`[Diarization] Starting acoustic diarization: ${audioPath}`);
    context.onProgress?.('Preparing acoustic speaker diarization...');

    const chunkBounds = this.normalizeChunkBounds(chunks);
    const normalizedOptions = this.normalizeOptions(context.options, chunkBounds.length);
    let speechSegments = context.speechSegments ?? [];
    if (speechSegments.length === 0 && chunkBounds.length >= 4) {
      try {
        context.onProgress?.('Preparing speech regions for classic diarization...');
        speechSegments = await this.getSpeechSegmentsForAudio(audioPath);
      } catch (error) {
        console.warn('[Diarization] Classic acoustic pass could not prepare VAD speech regions:', error);
      }
    }
    const passCandidates: Array<{ source: string; regions: AcousticRegion[] }> = [];
    const attemptedPasses: DiarizationDiagnostics["attemptedPasses"] = [];
    const embeddingCache = new Map<string, Promise<number[]>>();
    const speechRegions = this.buildAcousticRegions(chunkBounds, speechSegments);
    const vadChunkRegions = this.buildSpeechFocusedChunkRegions(chunkBounds, speechSegments);
    if (speechRegions.length > 0) {
      passCandidates.push({ source: "speech_region", regions: speechRegions });
    }
    if (normalizedOptions.preferVadBoundedRegions && vadChunkRegions.length > 0) {
      passCandidates.push({ source: "vad_chunk", regions: vadChunkRegions });
    }
    passCandidates.push({
      source: "chunk",
      regions: chunkBounds.map((bound, index) => {
        const clip = this.getChunkClip(chunkBounds, index);
        return { start: clip.start, end: clip.start + clip.duration, chunkIndices: [index] };
      }),
    });
    if (!normalizedOptions.preferVadBoundedRegions && vadChunkRegions.length > 0) {
      passCandidates.push({ source: "vad_chunk", regions: vadChunkRegions });
    }

    let finalPass: AcousticPassResult | null = null;
    let finalSource = passCandidates[0].source;

    for (let i = 0; i < passCandidates.length; i += 1) {
      this.throwIfAborted(context.signal);
      const candidate = passCandidates[i];
      context.onProgress?.(
        `Speaker diarization pass ${i + 1}/${passCandidates.length} (${candidate.source}, ${candidate.regions.length} regions)...`
      );
      const pass = await this.runAcousticPass(
        audioPath,
        chunkBounds,
        candidate.regions,
        normalizedOptions,
        embeddingCache,
        context.onProgress,
        context.signal
      );
      finalPass = pass;
      finalSource = candidate.source;
      attemptedPasses.push({
        source: candidate.source,
        regionCount: candidate.regions.length,
        uniqueSpeakerCount: pass.uniqueSpeakerCount,
        threshold: pass.threshold,
        quality: pass.quality,
        resegmentedChunkCount: pass.resegmentedChunkCount,
        resegmentationPasses: pass.resegmentationPasses,
      });

      console.log(
        `[Diarization] Acoustic clustering result: speakers=${pass.uniqueSpeakerCount}, regions=${candidate.regions.length}, source=${candidate.source}, threshold=${pass.threshold.toFixed(
          2
        )}, avgSim=${pass.quality.avgSimilarity.toFixed(4)}, p90Sim=${pass.quality.p90Similarity.toFixed(4)}, minSim=${pass.quality.minSimilarity.toFixed(4)}, maxSim=${pass.quality.maxSimilarity.toFixed(4)}`
      );

      const shouldTryNextCandidate =
        i < passCandidates.length - 1 &&
        candidate.source !== "chunk" &&
        pass.uniqueSpeakerCount <= 1 &&
        chunkBounds.length >= 6;
      if (shouldTryNextCandidate) {
        console.log(`[Diarization] ${candidate.source} acoustic pass collapsed to a single speaker, retrying with fallback regions.`);
        context.onProgress?.(`${candidate.source} diarization collapsed to one speaker, retrying with fallback regions...`);
        continue;
      }

      if (this.shouldFallbackToSemantic(pass.uniqueSpeakerCount, pass.quality, chunks.length)) {
        if (!normalizedOptions.semanticFallbackEnabled) {
          return {
            chunks: chunks.map((chunk, index) => ({
              ...chunk,
              speaker: `Speaker ${pass.assignedSpeakerIds[index] + 1}`,
            })),
            diagnostics: {
              provider: "acoustic",
              selectedSource: candidate.source as "speech_region" | "vad_chunk" | "chunk",
              speechSegmentCount: speechSegments.length,
              vadWindowCount: context.vadWindows?.length ?? 0,
              options: normalizedOptions,
              selectedPass: attemptedPasses[attemptedPasses.length - 1],
              attemptedPasses,
            },
          };
        }
        throw new DiarizationLowConfidenceError(
          `Acoustic diarization low confidence (source=${candidate.source}, speakers=${pass.uniqueSpeakerCount}, avgSim=${pass.quality.avgSimilarity.toFixed(
            4
          )}, p90Sim=${pass.quality.p90Similarity.toFixed(4)})`,
          {
            attemptedPasses,
            selectedPass: attemptedPasses[attemptedPasses.length - 1],
          }
        );
      }

      return {
        chunks: chunks.map((chunk, index) => ({
          ...chunk,
          speaker: `Speaker ${pass.assignedSpeakerIds[index] + 1}`,
        })),
        diagnostics: {
          provider: "acoustic",
          selectedSource: candidate.source as "speech_region" | "vad_chunk" | "chunk",
          speechSegmentCount: speechSegments.length,
          vadWindowCount: context.vadWindows?.length ?? 0,
          options: normalizedOptions,
          selectedPass: attemptedPasses[attemptedPasses.length - 1],
          attemptedPasses,
        },
      };
    }

    if (!finalPass) {
      return {
        chunks,
        diagnostics: {
          provider: "acoustic",
          selectedSource: "chunk",
          speechSegmentCount: speechSegments.length,
          vadWindowCount: context.vadWindows?.length ?? 0,
          options: normalizedOptions,
          attemptedPasses,
        },
      };
    }

    if (this.shouldFallbackToSemantic(finalPass.uniqueSpeakerCount, finalPass.quality, chunks.length)) {
      if (!normalizedOptions.semanticFallbackEnabled) {
        return {
          chunks: chunks.map((chunk, index) => ({
            ...chunk,
            speaker: `Speaker ${finalPass.assignedSpeakerIds[index] + 1}`,
          })),
          diagnostics: {
            provider: "acoustic",
            selectedSource: finalSource as "speech_region" | "vad_chunk" | "chunk",
            speechSegmentCount: speechSegments.length,
            vadWindowCount: context.vadWindows?.length ?? 0,
            options: normalizedOptions,
            selectedPass: attemptedPasses[attemptedPasses.length - 1],
            attemptedPasses,
          },
        };
      }
      throw new DiarizationLowConfidenceError(
        `Acoustic diarization low confidence (source=${finalSource}, speakers=${finalPass.uniqueSpeakerCount}, avgSim=${finalPass.quality.avgSimilarity.toFixed(
          4
        )}, p90Sim=${finalPass.quality.p90Similarity.toFixed(4)})`,
        {
          attemptedPasses,
          selectedPass: attemptedPasses[attemptedPasses.length - 1],
        }
      );
    }

    return {
      chunks: chunks.map((chunk, index) => ({
        ...chunk,
        speaker: `Speaker ${finalPass.assignedSpeakerIds[index] + 1}`,
      })),
      diagnostics: {
        provider: "acoustic",
        selectedSource: finalSource as "speech_region" | "vad_chunk" | "chunk",
        speechSegmentCount: speechSegments.length,
        vadWindowCount: context.vadWindows?.length ?? 0,
        options: normalizedOptions,
        selectedPass: attemptedPasses[attemptedPasses.length - 1],
        attemptedPasses,
      },
    };
  }

  private static async runAcousticPass(
    audioPath: string,
    chunkBounds: ChunkBounds[],
    regions: AcousticRegion[],
    options: NormalizedDiarizationOptions,
    embeddingCache: Map<string, Promise<number[]>>,
    onProgress?: (message: string) => void,
    signal?: AbortSignal
  ): Promise<AcousticPassResult> {
    const embeddings: number[][] = [];
    for (let i = 0; i < regions.length; i += 1) {
      this.throwIfAborted(signal);
      const region = regions[i];
      const embedding = await this.getRegionEmbedding(audioPath, region, embeddingCache);
      embeddings.push(embedding);

      if (i % 10 === 0 || i === regions.length - 1) {
        console.log(`[Diarization] Extracted speaker embedding ${i + 1}/${regions.length}`);
        onProgress?.(`Extracting speaker embeddings ${i + 1}/${regions.length}...`);
      }
    }

    const quality = this.measureEmbeddingQuality(embeddings);
    const clustering = this.clusterWithAdaptiveThreshold(embeddings, regions.length, quality, options);
    const mergedSpeakerIds = this.applySpeakerCountBias(
      embeddings,
      clustering.speakerIds,
      chunkBounds.length,
      options
    );
    const initialAssignedSpeakerIds = this.smoothAssignedSpeakerIds(
      this.assignChunkSpeakers(chunkBounds, regions, mergedSpeakerIds),
      options
    );
    const resegmented = this.resegmentAssignedSpeakerIds(chunkBounds, initialAssignedSpeakerIds, options);
    return {
      assignedSpeakerIds: resegmented.labels,
      uniqueSpeakerCount: new Set(resegmented.labels).size,
      quality,
      threshold: clustering.threshold,
      rawSpeakerIds: clustering.speakerIds,
      postMergeSpeakerIds: mergedSpeakerIds,
      resegmentedChunkCount: resegmented.changedChunkCount,
      resegmentationPasses: resegmented.passes,
    };
  }

  private static getRegionCacheKey(start: number, end: number) {
    return `${start.toFixed(3)}:${end.toFixed(3)}`;
  }

  private static async getRegionEmbedding(
    audioPath: string,
    region: AcousticRegion,
    embeddingCache: Map<string, Promise<number[]>>
  ) {
    const key = this.getRegionCacheKey(region.start, region.end);
    let pending = embeddingCache.get(key);
    if (!pending) {
      pending = (async () => {
        const samples = await this.extractAudioClip(audioPath, region.start, region.end - region.start);
        return SpeakerEmbeddingService.getEmbedding(samples);
      })();
      embeddingCache.set(key, pending);
    }

    try {
      return await pending;
    } catch (error) {
      embeddingCache.delete(key);
      throw error;
    }
  }

  private static shouldFallbackToSemantic(
    uniqueSpeakerCount: number,
    quality: EmbeddingQuality,
    chunkCount: number
  ) {
    if (chunkCount < 4 || quality.pairCount === 0) return false;
    if (uniqueSpeakerCount > 1 && uniqueSpeakerCount <= this.maxAssignedSpeakerCount) return false;
    return (
      uniqueSpeakerCount > this.maxAssignedSpeakerCount ||
      quality.p90Similarity >= this.acousticLowConfidenceP90 ||
      quality.avgSimilarity >= this.acousticLowConfidenceAvg
    );
  }

  private static clusterWithAdaptiveThreshold(
    embeddings: number[][],
    chunkCount: number,
    quality: EmbeddingQuality,
    options: NormalizedDiarizationOptions
  ) {
    const preferredThreshold = this.resolveClusterThreshold(quality);
    const thresholds: number[] = [];
    for (let step = this.acousticClusterThresholdMin; step <= this.acousticClusterThresholdMax + 0.001; step += 0.025) {
      thresholds.push(Number(step.toFixed(3)));
    }
    thresholds.sort((a, b) => Math.abs(a - preferredThreshold) - Math.abs(b - preferredThreshold));

    let bestThreshold = preferredThreshold;
    let bestSpeakerIds = ClusteringService.cluster(embeddings, preferredThreshold);
    let bestScore = this.scoreClusterCount(new Set(bestSpeakerIds).size, preferredThreshold, preferredThreshold, options, chunkCount);

    for (const threshold of thresholds) {
      const speakerIds = ClusteringService.cluster(embeddings, threshold);
      const uniqueSpeakerCount = new Set(speakerIds).size;
      const score = this.scoreClusterCount(uniqueSpeakerCount, threshold, preferredThreshold, options, chunkCount);
      if (score < bestScore) {
        bestScore = score;
        bestThreshold = threshold;
        bestSpeakerIds = speakerIds;
      }
    }

    return { speakerIds: bestSpeakerIds, threshold: bestThreshold };
  }

  private static resolveClusterThreshold(quality: EmbeddingQuality) {
    const proposed = quality.avgSimilarity + 0.1;
    return Math.min(this.acousticClusterThresholdMax, Math.max(this.acousticClusterThresholdMin, proposed));
  }

  private static normalizeChunkBounds(chunks: TranscribedChunk[]): ChunkBounds[] {
    return chunks.map((chunk, index) => {
      const start = Number.isFinite(chunk.start_ts) ? chunk.start_ts : index * this.minClipDurationSec;
      const nextStart =
        index < chunks.length - 1 && Number.isFinite(chunks[index + 1].start_ts)
          ? chunks[index + 1].start_ts
          : start + this.minClipDurationSec;
      const rawEnd =
        typeof chunk.end_ts === "number" && Number.isFinite(chunk.end_ts) && chunk.end_ts > start
          ? chunk.end_ts
          : Math.max(start + this.minClipDurationSec, nextStart);
      return {
        index,
        start,
        end: Math.max(rawEnd, start + 0.1),
      };
    });
  }

  private static buildAcousticRegions(
    chunkBounds: ChunkBounds[],
    speechSegments?: Array<{ start: number; end: number }>
  ): AcousticRegion[] {
    const normalizedSegments = (speechSegments || [])
      .map((segment) => ({
        start: Number(segment?.start),
        end: Number(segment?.end),
      }))
      .filter((segment) => Number.isFinite(segment.start) && Number.isFinite(segment.end) && segment.end > segment.start)
      .sort((a, b) => a.start - b.start);

    if (normalizedSegments.length === 0) return [];

    const mergedSegments: Array<{ start: number; end: number }> = [];
    for (const segment of normalizedSegments) {
      const previous = mergedSegments[mergedSegments.length - 1];
      if (
        previous &&
        segment.start - previous.end <= this.regionMergeGapSec &&
        segment.end - previous.start <= this.maxClipDurationSec
      ) {
        previous.end = Math.max(previous.end, segment.end);
        continue;
      }
      mergedSegments.push({ ...segment });
    }

    const regions: AcousticRegion[] = [];
    let current: AcousticRegion | null = null;
    for (const segment of mergedSegments) {
      if (!current) {
        current = {
          start: segment.start,
          end: segment.end,
          chunkIndices: this.findChunkIndicesForRegion(chunkBounds, segment.start, segment.end),
        };
        continue;
      }

      const mergedDuration = segment.end - current.start;
      const sameNeighborhood = segment.start - current.end <= this.regionMergeGapSec;
      if (sameNeighborhood && mergedDuration <= this.targetRegionDurationSec) {
        current.end = Math.max(current.end, segment.end);
        current.chunkIndices = this.mergeIndices(
          current.chunkIndices,
          this.findChunkIndicesForRegion(chunkBounds, current.start, current.end)
        );
      } else {
        if (current.chunkIndices.length > 0) regions.push(current);
        current = {
          start: segment.start,
          end: segment.end,
          chunkIndices: this.findChunkIndicesForRegion(chunkBounds, segment.start, segment.end),
        };
      }
    }

    if (current?.chunkIndices.length) {
      regions.push(current);
    }

    return regions;
  }

  private static buildSpeechFocusedChunkRegions(
    chunkBounds: ChunkBounds[],
    speechSegments?: Array<{ start: number; end: number }>
  ): AcousticRegion[] {
    const normalizedSegments = (speechSegments || [])
      .map((segment) => ({
        start: Number(segment?.start),
        end: Number(segment?.end),
      }))
      .filter((segment) => Number.isFinite(segment.start) && Number.isFinite(segment.end) && segment.end > segment.start)
      .sort((a, b) => a.start - b.start);

    if (normalizedSegments.length === 0) return [];

    return chunkBounds.map((chunk) => {
      const paddedStart = Math.max(0, chunk.start - this.regionAssignPaddingSec);
      const paddedEnd = chunk.end + this.regionAssignPaddingSec;
      const overlaps = normalizedSegments.filter(
        (segment) => this.overlapDuration(segment.start, segment.end, paddedStart, paddedEnd) > 0
      );

      if (overlaps.length === 0) {
        const clip = this.getChunkClip(chunkBounds, chunk.index);
        return { start: clip.start, end: clip.start + clip.duration, chunkIndices: [chunk.index] };
      }

      let start = overlaps[0].start;
      let end = overlaps[overlaps.length - 1].end;
      const midpoint = chunk.start + (chunk.end - chunk.start) / 2;
      const currentDuration = end - start;

      if (currentDuration < this.minClipDurationSec) {
        start = Math.max(0, midpoint - this.minClipDurationSec / 2);
        end = start + this.minClipDurationSec;
      }

      if (end - start > this.maxClipDurationSec) {
        start = Math.max(0, midpoint - this.maxClipDurationSec / 2);
        end = start + this.maxClipDurationSec;
      }

      return { start, end, chunkIndices: [chunk.index] };
    });
  }

  private static findChunkIndicesForRegion(chunkBounds: ChunkBounds[], start: number, end: number) {
    const directMatches = chunkBounds
      .filter((chunk) => this.overlapDuration(chunk.start, chunk.end, start, end) >= this.regionAssignOverlapMinSec)
      .map((chunk) => chunk.index);

    if (directMatches.length > 0) return directMatches;

    const paddedStart = Math.max(0, start - this.regionAssignPaddingSec);
    const paddedEnd = end + this.regionAssignPaddingSec;
    return chunkBounds
      .filter((chunk) => this.overlapDuration(chunk.start, chunk.end, paddedStart, paddedEnd) > 0)
      .map((chunk) => chunk.index);
  }

  private static mergeIndices(a: number[], b: number[]) {
    return Array.from(new Set([...a, ...b])).sort((x, y) => x - y);
  }

  private static measureEmbeddingQuality(embeddings: number[][]): EmbeddingQuality {
    const similarities: number[] = [];
    for (let i = 0; i < embeddings.length; i += 1) {
      for (let j = i + 1; j < embeddings.length; j += 1) {
        similarities.push(ClusteringService.cosineSimilarity(embeddings[i], embeddings[j]));
      }
    }

    if (similarities.length === 0) {
      return {
        pairCount: 0,
        minSimilarity: 0,
        maxSimilarity: 0,
        avgSimilarity: 0,
        p90Similarity: 0,
      };
    }

    similarities.sort((a, b) => a - b);
    const sum = similarities.reduce((acc, value) => acc + value, 0);
    const p90Index = Math.min(similarities.length - 1, Math.floor(similarities.length * 0.9));
    return {
      pairCount: similarities.length,
      minSimilarity: similarities[0],
      maxSimilarity: similarities[similarities.length - 1],
      avgSimilarity: sum / similarities.length,
      p90Similarity: similarities[p90Index],
    };
  }

  private static assignChunkSpeakers(
    chunkBounds: ChunkBounds[],
    regions: AcousticRegion[],
    regionSpeakerIds: number[]
  ) {
    return chunkBounds.map((chunk) => {
      const weightedScores = new Map<number, number>();
      regions.forEach((region, index) => {
        const overlap = this.overlapDuration(chunk.start, chunk.end, region.start, region.end);
        if (overlap > 0) {
          const speakerId = regionSpeakerIds[index];
          weightedScores.set(speakerId, (weightedScores.get(speakerId) || 0) + overlap);
        }
      });

      if (weightedScores.size > 0) {
        return Array.from(weightedScores.entries()).sort((a, b) => b[1] - a[1])[0][0];
      }

      let nearestIndex = 0;
      let nearestDistance = Number.POSITIVE_INFINITY;
      const midpoint = chunk.start + (chunk.end - chunk.start) / 2;
      regions.forEach((region, index) => {
        const regionMidpoint = region.start + (region.end - region.start) / 2;
        const distance = Math.abs(regionMidpoint - midpoint);
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearestIndex = index;
        }
      });
      return regionSpeakerIds[nearestIndex] ?? 0;
    });
  }

  private static smoothAssignedSpeakerIds(
    assignedSpeakerIds: number[],
    options: NormalizedDiarizationOptions
  ) {
    if (assignedSpeakerIds.length < 3) return assignedSpeakerIds;

    const counts = new Map<number, number>();
    assignedSpeakerIds.forEach((speakerId) => {
      counts.set(speakerId, (counts.get(speakerId) || 0) + 1);
    });

    const smoothed = [...assignedSpeakerIds];
    for (let i = 0; i < smoothed.length; i += 1) {
      const speakerId = smoothed[i];
      const maxSingletonSize = options.allowShortInterjectionSpeaker ? 0 : 1;
      if ((counts.get(speakerId) || 0) > maxSingletonSize) continue;

      const prev = i > 0 ? smoothed[i - 1] : null;
      const next = i < smoothed.length - 1 ? smoothed[i + 1] : null;
      if (prev != null && next != null && prev === next) {
        smoothed[i] = prev;
        continue;
      }

      const candidates = [prev, next].filter((value): value is number => value != null);
      if (candidates.length === 0) continue;

      candidates.sort((a, b) => (counts.get(b) || 0) - (counts.get(a) || 0));
      smoothed[i] = candidates[0];
    }

    return smoothed;
  }

  private static resegmentAssignedSpeakerIds(
    chunkBounds: ChunkBounds[],
    assignedSpeakerIds: number[],
    options: NormalizedDiarizationOptions
  ) {
    if (assignedSpeakerIds.length < 3) {
      return { labels: assignedSpeakerIds, changedChunkCount: 0, passes: 0 };
    }

    let labels = [...assignedSpeakerIds];
    let totalChanged = 0;
    let passes = 0;

    for (let passIndex = 0; passIndex < 3; passIndex += 1) {
      let changedThisPass = 0;
      const dominantSpeakerId = options.preferStablePrimarySpeaker
        ? this.findDominantSpeakerId(labels, chunkBounds)
        : null;

      let runs = this.buildSpeakerRuns(labels, chunkBounds);

      for (let i = 1; i < runs.length - 1; i += 1) {
        const prev = runs[i - 1];
        const current = runs[i];
        const next = runs[i + 1];
        if (prev.speakerId !== next.speakerId) continue;
        if (!this.shouldResegmentRun(current, options, true)) continue;
        changedThisPass += this.replaceSpeakerRun(labels, current, prev.speakerId);
      }

      if (changedThisPass === 0) {
        runs = this.buildSpeakerRuns(labels, chunkBounds);
        for (let i = 0; i < runs.length; i += 1) {
          const current = runs[i];
          if (!this.shouldResegmentRun(current, options, false)) continue;
          const prev = i > 0 ? runs[i - 1] : null;
          const next = i < runs.length - 1 ? runs[i + 1] : null;
          const targetSpeakerId = this.pickResegmentationTarget(current, prev, next, dominantSpeakerId, options);
          if (targetSpeakerId == null || targetSpeakerId === current.speakerId) continue;
          changedThisPass += this.replaceSpeakerRun(labels, current, targetSpeakerId);
        }
      }

      if (changedThisPass === 0) break;
      totalChanged += changedThisPass;
      passes += 1;
    }

    return { labels, changedChunkCount: totalChanged, passes };
  }

  private static shouldResegmentRun(
    run: { speakerId: number; startIndex: number; endIndex: number; chunkCount: number; duration: number },
    options: NormalizedDiarizationOptions,
    isBridgePattern: boolean
  ) {
    const baseMaxDuration = options.allowShortInterjectionSpeaker ? 0.7 : 1.5;
    const exactTwoSpeakerMode =
      options.exactSpeakerCount === 2 || (options.minSpeakers <= 2 && options.maxSpeakers <= 2);
    const maxDuration = isBridgePattern
      ? baseMaxDuration + (exactTwoSpeakerMode ? 0.8 : 0.3)
      : baseMaxDuration + (exactTwoSpeakerMode ? 0.4 : 0);
    const maxChunkCount = isBridgePattern
      ? (options.allowShortInterjectionSpeaker ? 1 : exactTwoSpeakerMode ? 3 : 2)
      : (options.allowShortInterjectionSpeaker ? 1 : 2);

    if (run.duration <= maxDuration) return true;
    if (run.chunkCount <= maxChunkCount && run.duration <= maxDuration * 1.4) return true;
    return false;
  }

  private static findDominantSpeakerId(labels: number[], chunkBounds: ChunkBounds[]) {
    const durations = new Map<number, number>();
    labels.forEach((speakerId, index) => {
      const chunk = chunkBounds[index];
      const duration = Math.max(0.1, chunk.end - chunk.start);
      durations.set(speakerId, (durations.get(speakerId) || 0) + duration);
    });

    let dominantSpeakerId: number | null = null;
    let maxDuration = -Infinity;
    for (const [speakerId, duration] of durations.entries()) {
      if (duration > maxDuration) {
        maxDuration = duration;
        dominantSpeakerId = speakerId;
      }
    }
    return dominantSpeakerId;
  }

  private static buildSpeakerRuns(labels: number[], chunkBounds: ChunkBounds[]) {
    const runs: Array<{
      speakerId: number;
      startIndex: number;
      endIndex: number;
      chunkCount: number;
      duration: number;
    }> = [];

    let startIndex = 0;
    for (let i = 1; i <= labels.length; i += 1) {
      if (i < labels.length && labels[i] === labels[startIndex]) continue;

      const endIndex = i - 1;
      let duration = 0;
      for (let j = startIndex; j <= endIndex; j += 1) {
        duration += Math.max(0.1, chunkBounds[j].end - chunkBounds[j].start);
      }
      runs.push({
        speakerId: labels[startIndex],
        startIndex,
        endIndex,
        chunkCount: endIndex - startIndex + 1,
        duration,
      });
      startIndex = i;
    }

    return runs;
  }

  private static pickResegmentationTarget(
    current: { speakerId: number; startIndex: number; endIndex: number; chunkCount: number; duration: number },
    prev: { speakerId: number; startIndex: number; endIndex: number; chunkCount: number; duration: number } | null,
    next: { speakerId: number; startIndex: number; endIndex: number; chunkCount: number; duration: number } | null,
    dominantSpeakerId: number | null,
    options: NormalizedDiarizationOptions
  ) {
    if (prev && next && prev.speakerId === next.speakerId) {
      return prev.speakerId;
    }

    const exactTwoSpeakerMode =
      options.exactSpeakerCount === 2 || (options.minSpeakers <= 2 && options.maxSpeakers <= 2);

    if (dominantSpeakerId != null && current.duration <= (exactTwoSpeakerMode ? 2.2 : 1.4)) {
      if ((prev && prev.speakerId === dominantSpeakerId) || (next && next.speakerId === dominantSpeakerId)) {
        return dominantSpeakerId;
      }
    }

    const candidates = [prev, next].filter(
      (run): run is { speakerId: number; startIndex: number; endIndex: number; chunkCount: number; duration: number } =>
        run != null
    );
    if (candidates.length === 0) return null;

    candidates.sort((a, b) => {
      const dominantBoostA = a.speakerId === dominantSpeakerId ? 0.6 : 0;
      const dominantBoostB = b.speakerId === dominantSpeakerId ? 0.6 : 0;
      const scoreA = a.duration + a.chunkCount * 0.15 + dominantBoostA;
      const scoreB = b.duration + b.chunkCount * 0.15 + dominantBoostB;
      return scoreB - scoreA;
    });
    return candidates[0].speakerId;
  }

  private static replaceSpeakerRun(
    labels: number[],
    run: { startIndex: number; endIndex: number },
    nextSpeakerId: number
  ) {
    let changed = 0;
    for (let index = run.startIndex; index <= run.endIndex; index += 1) {
      if (labels[index] === nextSpeakerId) continue;
      labels[index] = nextSpeakerId;
      changed += 1;
    }
    return changed;
  }

  private static applySpeakerCountBias(
    embeddings: number[][],
    speakerIds: number[],
    chunkCount: number,
    options: NormalizedDiarizationOptions
  ) {
    if (chunkCount < 8) return speakerIds;
    if (!options.forceMergeTinyClustersInTwoSpeakerMode) return speakerIds;
    if (options.exactSpeakerCount !== 2 && !(options.minSpeakers <= 2 && options.maxSpeakers <= 2)) {
      return speakerIds;
    }
    const uniqueSpeakerCount = new Set(speakerIds).size;
    if (uniqueSpeakerCount !== 3) return speakerIds;

    const counts = new Map<number, number>();
    speakerIds.forEach((speakerId) => counts.set(speakerId, (counts.get(speakerId) || 0) + 1));
    const sortedCounts = [...counts.entries()].sort((a, b) => b[1] - a[1]);
    const coveredByTopTwo = ((sortedCounts[0]?.[1] || 0) + (sortedCounts[1]?.[1] || 0)) / speakerIds.length;
    const smallestClusterSize = sortedCounts[2]?.[1] || 0;

    if (smallestClusterSize > 1 || coveredByTopTwo < 0.8) {
      return speakerIds;
    }

    return ClusteringService.mergeTinyClusters(embeddings, speakerIds, {
      maxClusters: 2,
      minClusterSize: 1,
      minCoverageRatio: 0.15,
    });
  }

  private static normalizeOptions(options: DiarizationOptions | undefined, chunkCount: number): NormalizedDiarizationOptions {
    const provider = options?.provider === 'pyannote' ? 'pyannote' : 'classic';
    const mode = options?.mode || 'auto';
    const scenePreset = options?.scenePreset || 'interview';
    let exactSpeakerCount = Number.isFinite(options?.exactSpeakerCount as number) ? Math.round(options!.exactSpeakerCount as number) : null;
    let minSpeakers = Number.isFinite(options?.minSpeakers as number) ? Math.round(options!.minSpeakers as number) : 1;
    let maxSpeakers = Number.isFinite(options?.maxSpeakers as number) ? Math.round(options!.maxSpeakers as number) : Math.min(6, Math.max(2, Math.ceil(chunkCount / 4)));

    if (mode === 'fixed' && !exactSpeakerCount) {
      exactSpeakerCount = scenePreset === 'interview' ? 2 : 2;
    }
    if (mode === 'fixed' && exactSpeakerCount) {
      minSpeakers = exactSpeakerCount;
      maxSpeakers = exactSpeakerCount;
    } else if (mode === 'range') {
      minSpeakers = Math.max(1, minSpeakers);
      maxSpeakers = Math.max(minSpeakers, maxSpeakers);
    } else if (mode === 'many') {
      minSpeakers = Math.max(2, minSpeakers);
      maxSpeakers = Math.max(4, maxSpeakers);
    } else if (mode === 'auto') {
      if (scenePreset === 'interview') {
        minSpeakers = 2;
        maxSpeakers = 2;
        exactSpeakerCount = 2;
      } else if (scenePreset === 'podcast') {
        minSpeakers = 2;
        maxSpeakers = 4;
      } else if (scenePreset === 'presentation_qa') {
        minSpeakers = 2;
        maxSpeakers = 4;
      } else if (scenePreset === 'meeting') {
        minSpeakers = 2;
        maxSpeakers = Math.max(6, maxSpeakers);
      }
    }

    minSpeakers = Math.max(1, minSpeakers);
    maxSpeakers = Math.max(minSpeakers, maxSpeakers);

    return {
      provider,
      mode,
      exactSpeakerCount,
      minSpeakers,
      maxSpeakers,
      scenePreset,
      preferStablePrimarySpeaker:
        options?.preferStablePrimarySpeaker ?? scenePreset !== 'meeting',
      allowShortInterjectionSpeaker:
        options?.allowShortInterjectionSpeaker ?? scenePreset === 'meeting',
      preferVadBoundedRegions:
        options?.preferVadBoundedRegions ?? scenePreset !== 'meeting',
      forceMergeTinyClustersInTwoSpeakerMode:
        options?.forceMergeTinyClustersInTwoSpeakerMode ?? (scenePreset === 'interview' || exactSpeakerCount === 2),
      semanticFallbackEnabled:
        options?.semanticFallbackEnabled ?? true,
    };
  }

  private static assignTurnsToChunkSpeakers(
    chunkBounds: ChunkBounds[],
    turns: Array<{ start: number; end: number; speaker: string }>
  ) {
    if (turns.length === 0) {
      return chunkBounds.map(() => 'Speaker 1');
    }

    return chunkBounds.map((chunk) => {
      let bestSpeaker = turns[0].speaker;
      let bestOverlap = -1;
      for (const turn of turns) {
        const overlap = this.overlapDuration(chunk.start, chunk.end, turn.start, turn.end);
        if (overlap > bestOverlap) {
          bestOverlap = overlap;
          bestSpeaker = turn.speaker;
        }
      }

      if (bestOverlap > 0) return bestSpeaker;

      const chunkMidpoint = (chunk.start + chunk.end) / 2;
      let closestSpeaker = turns[0].speaker;
      let closestDistance = Number.POSITIVE_INFINITY;
      for (const turn of turns) {
        const turnMidpoint = (turn.start + turn.end) / 2;
        const distance = Math.abs(chunkMidpoint - turnMidpoint);
        if (distance < closestDistance) {
          closestDistance = distance;
          closestSpeaker = turn.speaker;
        }
      }
      return closestSpeaker;
    });
  }

  private static scoreClusterCount(
    uniqueSpeakerCount: number,
    threshold: number,
    preferredThreshold: number,
    options: NormalizedDiarizationOptions,
    chunkCount: number
  ) {
    let penalty = Math.abs(threshold - preferredThreshold);
    const cappedClusterCount = Math.max(1, uniqueSpeakerCount);

    if (options.exactSpeakerCount) {
      penalty += Math.abs(cappedClusterCount - options.exactSpeakerCount) * 5;
    } else {
      if (cappedClusterCount < options.minSpeakers) {
        penalty += (options.minSpeakers - cappedClusterCount) * 5;
      }
      if (cappedClusterCount > options.maxSpeakers) {
        penalty += (cappedClusterCount - options.maxSpeakers) * 5;
      }
    }

    if (options.preferStablePrimarySpeaker && cappedClusterCount > 2 && chunkCount >= 8) {
      penalty += (cappedClusterCount - 2) * 0.75;
    }
    if (!options.allowShortInterjectionSpeaker && cappedClusterCount > 3) {
      penalty += (cappedClusterCount - 3) * 0.5;
    }

    return penalty;
  }

  private static overlapDuration(aStart: number, aEnd: number, bStart: number, bEnd: number) {
    return Math.max(0, Math.min(aEnd, bEnd) - Math.max(aStart, bStart));
  }

  private static getChunkClip(chunkBounds: ChunkBounds[], index: number) {
    const chunk = chunkBounds[index];
    const rawDuration = Math.max(0.1, chunk.end - chunk.start);
    const targetDuration = Math.min(
      this.maxClipDurationSec,
      Math.max(this.minClipDurationSec, rawDuration)
    );
    const midpoint = chunk.start + rawDuration / 2;
    const previousBoundary = index > 0 ? chunkBounds[index - 1].end : 0;
    const nextBoundary =
      index < chunkBounds.length - 1
        ? chunkBounds[index + 1].start
        : midpoint + targetDuration / 2;

    let start = Math.max(0, midpoint - targetDuration / 2);
    let end = start + targetDuration;

    if (previousBoundary > 0) {
      start = Math.max(start, previousBoundary - 0.15);
    }
    end = Math.max(end, start + this.minClipDurationSec);
    end = Math.min(end, nextBoundary + 0.15, start + this.maxClipDurationSec);

    if (end - start < this.minClipDurationSec) {
      end = start + this.minClipDurationSec;
    }

    return { start, duration: end - start };
  }

  private static async performSemanticDiarization(chunks: TranscribedChunk[]): Promise<TranscribedChunk[]> {
    console.log(`[Diarization] Starting semantic diarization for ${chunks.length} chunks`);

    const settings = await SettingsManager.getSettings({ mask: false });
    const geminiModel = settings.translateModels.find(
      (model) => model.name.toLowerCase().includes("gemini") || model.id.toLowerCase().includes("gemini")
    );

    if (!geminiModel?.key) {
      console.warn("[Diarization] Gemini API key not configured. Returning chunks without speaker tags.");
      return chunks;
    }

    const ai = new GoogleGenAI({ apiKey: geminiModel.key });
    const textToAnalyze = chunks.map((chunk, index) => `${index}: ${chunk.text}`).join("\n");
    const systemInstruction = [
      "Assign speaker labels to each subtitle line.",
      "Use compact labels such as Speaker 1, Speaker 2, Speaker 3.",
      `Return a JSON array with exactly ${chunks.length} strings.`,
      "Do not include explanations.",
    ].join(" ");

    try {
      const response = await ai.models.generateContent({
        model: "gemini-1.5-flash",
        contents: `Assign speaker labels to these lines:\n${textToAnalyze}`,
        config: { systemInstruction, temperature: 0.1 },
      });

      const speakerTags = JSON.parse(response.text.replace(/```json|```/g, "").trim());
      if (Array.isArray(speakerTags) && speakerTags.length === chunks.length) {
        return chunks.map((chunk, index) => ({
          ...chunk,
          speaker: String(speakerTags[index]),
        }));
      }
    } catch (error) {
      console.error("[Diarization] Semantic diarization failed:", error);
    }

    return chunks;
  }

  private static async extractAudioClip(filePath: string, start: number, duration: number): Promise<Float32Array> {
    return new Promise((resolve, reject) => {
      const ffmpeg = spawn(this.getFfmpegBinary(), [
        "-v",
        "error",
        "-ss",
        start.toString(),
        "-t",
        duration.toString(),
        "-i",
        filePath,
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "pipe:1",
      ]);

      const stdoutChunks: Buffer[] = [];
      const stderrChunks: Buffer[] = [];
      ffmpeg.stdout.on("data", (chunk: Buffer) => stdoutChunks.push(chunk));
      ffmpeg.stderr.on("data", (chunk: Buffer) => stderrChunks.push(chunk));

      ffmpeg.on("close", (code) => {
        if (code === 0) {
          const buffer = Buffer.concat(stdoutChunks);
          resolve(new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4));
          return;
        }

        const stderr = Buffer.concat(stderrChunks).toString("utf8").trim();
        reject(new Error(stderr || "FFmpeg clip extraction failed"));
      });
    });
  }
}
