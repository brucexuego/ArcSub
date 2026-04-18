/**
 * Agglomerative Hierarchical Clustering (AHC) for Audio Diarization
 */
export class ClusteringService {
  /**
   * Performs clustering on a list of embeddings
   * @param embeddings Array of N-dimensional feature vectors (e.g. 192 from ECAPA-TDNN)
   * @param threshold Cosine similarity threshold (0-1, higher means more precise/more speakers)
   * @returns Array of group IDs (one for each input embedding)
   */
  static cluster(embeddings: number[][], threshold: number = 0.5): number[] {
    if (embeddings.length === 0) return [];
    if (embeddings.length === 1) return [0];

    // Initial state: Each embedding is its own cluster
    const clusters = embeddings.map((vec, i) => ({
      indices: [i],
      center: vec
    }));

    let merged = true;
    while (merged && clusters.length > 1) {
      merged = false;
      let maxSim = -Infinity;
      let clusterA = -1;
      let clusterB = -1;

      // Find the most similar pair
      for (let i = 0; i < clusters.length; i++) {
        for (let j = i + 1; j < clusters.length; j++) {
          const sim = this.cosineSimilarity(clusters[i].center, clusters[j].center);
          if (sim > maxSim) {
            maxSim = sim;
            clusterA = i;
            clusterB = j;
          }
        }
      }

      // If similarity exceeds threshold, merge them
      if (maxSim > threshold) {
        // Merge clusterB into clusterA
        clusters[clusterA].indices.push(...clusters[clusterB].indices);
        clusters[clusterA].center = this.calculateAverage(
          clusters[clusterA].center,
          clusters[clusterB].center,
          clusters[clusterA].indices.length - clusters[clusterB].indices.length,
          clusters[clusterB].indices.length
        );
        clusters.splice(clusterB, 1);
        merged = true;
      }
    }

    // Map initial index back to group ID
    const result = new Array(embeddings.length);
    clusters.forEach((cluster, groupId) => {
      cluster.indices.forEach(idx => {
        result[idx] = groupId;
      });
    });

    return result;
  }

  static mergeTinyClusters(
    embeddings: number[][],
    labels: number[],
    options: {
      maxClusters: number;
      minClusterSize?: number;
      minCoverageRatio?: number;
    }
  ): number[] {
    if (embeddings.length === 0 || labels.length !== embeddings.length) return labels;

    const maxClusters = Math.max(1, Math.floor(options.maxClusters));
    const minClusterSize = Math.max(1, Math.floor(options.minClusterSize ?? 1));
    const minCoverageRatio = Math.min(1, Math.max(0, options.minCoverageRatio ?? 0));
    let normalized = this.normalizeLabels(labels);

    while (new Set(normalized).size > maxClusters) {
      const stats = this.buildClusterStats(embeddings, normalized);
      const sorted = [...stats.values()].sort((a, b) => a.indices.length - b.indices.length);
      const smallest = sorted[0];
      if (!smallest) break;

      const coverage = smallest.indices.length / normalized.length;
      if (smallest.indices.length > minClusterSize && coverage > minCoverageRatio) {
        break;
      }

      let bestTargetId = -1;
      let bestSimilarity = -Infinity;
      for (const candidate of sorted.slice(1)) {
        const similarity = this.cosineSimilarity(smallest.center, candidate.center);
        if (similarity > bestSimilarity) {
          bestSimilarity = similarity;
          bestTargetId = candidate.id;
        }
      }

      if (bestTargetId < 0) break;
      normalized = normalized.map((label) => (label === smallest.id ? bestTargetId : label));
      normalized = this.normalizeLabels(normalized);
    }

    return normalized;
  }

  static cosineSimilarity(vecA: number[], vecB: number[]): number {
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vecA.length; i++) {
      dot += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  private static calculateAverage(centerA: number[], centerB: number[], weightA: number, weightB: number): number[] {
    const total = weightA + weightB;
    return centerA.map((val, i) => (val * weightA + centerB[i] * weightB) / total);
  }

  private static normalizeLabels(labels: number[]): number[] {
    const remap = new Map<number, number>();
    let nextId = 0;
    return labels.map((label) => {
      if (!remap.has(label)) {
        remap.set(label, nextId++);
      }
      return remap.get(label)!;
    });
  }

  private static buildClusterStats(embeddings: number[][], labels: number[]) {
    const stats = new Map<number, { id: number; indices: number[]; center: number[] }>();
    labels.forEach((label, index) => {
      const existing = stats.get(label);
      if (!existing) {
        stats.set(label, { id: label, indices: [index], center: [...embeddings[index]] });
        return;
      }
      const countBefore = existing.indices.length;
      existing.indices.push(index);
      existing.center = this.calculateAverage(existing.center, embeddings[index], countBefore, 1);
    });
    return stats;
  }
}
