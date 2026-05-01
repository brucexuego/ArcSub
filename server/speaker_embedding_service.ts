import path from "path";
import fs from "fs-extra";
import { PathManager } from "./path_manager.js";
import { AudioProcessor } from "./audio_processor.js";
import { OpenvinoBackend } from "./openvino_backend.js";
import { ResourceManager } from "./services/resource_manager.js";

export class SpeakerEmbeddingService {
  private static compiledModel: any | null = null;
  private static inputName: string | null = null;
  private static modelPath = path.join(PathManager.getModelsPath(), "ecapa-tdnn.onnx");
  private static irModelPath = path.join(PathManager.getModelsPath(), "ecapa-tdnn.xml");

  private static async getPreferredModelPath() {
    if (await fs.pathExists(this.irModelPath)) {
      return this.irModelPath;
    }
    return this.modelPath;
  }

  /**
   * Initializes the OpenVINO compiled model on AUTO device selection.
   */
  static async init() {
    if (this.compiledModel) return;
    const hasAnyModel = (await fs.pathExists(this.irModelPath)) || (await fs.pathExists(this.modelPath));
    if (!hasAnyModel) {
      try {
        await ResourceManager.ensureBaselineAsset("speaker_embedding");
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        console.warn(`[Embedding] Could not download speaker embedding model, continuing with existing fallback behavior: ${message}`);
      }
    }

    const modelPath = await this.getPreferredModelPath();
    if (!(await fs.pathExists(modelPath))) {
      throw new Error(`[Embedding] model not found: ${this.irModelPath} or ${this.modelPath}`);
    }

    const device = OpenvinoBackend.getBaselineModelDevice();
    console.log(`[Embedding] Compiling model with OpenVINO device: ${device} (${path.basename(modelPath)})`);
    this.compiledModel = await OpenvinoBackend.compileModel(modelPath, device);
    this.inputName = OpenvinoBackend.getPortName(this.compiledModel.input(0), "input");
    console.log("[Embedding] OpenVINO model ready");
  }

  /**
   * Extract embedding vector using OpenVINO runtime.
   */
  static async getEmbedding(samples: Float32Array): Promise<number[]> {
    await this.init();
    if (!this.compiledModel || !this.inputName) throw new Error("Embedding compiled model missing");

    try {
      const rawFeatures = await AudioProcessor.extractFeatures(samples);
      const currentFrames = rawFeatures.length / 80;
      const targetFrames = 360;

      let finalFeatures: Float32Array;
      if (currentFrames < targetFrames) {
        finalFeatures = new Float32Array(targetFrames * 80);
        finalFeatures.set(rawFeatures);
      } else {
        finalFeatures = rawFeatures.slice(0, targetFrames * 80);
      }

      const tensor = await OpenvinoBackend.createTensor("f32", [1, targetFrames, 80], finalFeatures);
      const inferRequest = this.compiledModel.createInferRequest();
      const results = inferRequest.infer({ [this.inputName]: tensor });
      const output = OpenvinoBackend.getTensorData(Object.values(results)[0]) as Float32Array | null;
      if (!output) {
        throw new Error("Embedding output tensor is empty.");
      }
      return Array.from(output);
    } catch (err: any) {
      console.error("[Embedding] inference failed:", err?.message || err);
      return new Array(192).fill(0);
    }
  }
}
