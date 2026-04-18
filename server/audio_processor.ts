/**
 * Audio feature extraction for speaker embeddings.
 *
 * This is still a lightweight JS implementation, but it is closer to the
 * standard ECAPA/SpeechBrain frontend than the previous version:
 * - pre-emphasis
 * - 25 ms window / 10 ms hop at 16 kHz
 * - power spectrum instead of magnitude spectrum
 * - log Mel filterbank features
 * - per-bin mean normalization across time
 */
export class AudioProcessor {
  private static readonly SAMPLE_RATE = 16000;
  private static readonly N_FFT = 512;
  private static readonly WIN_LENGTH = 400;
  private static readonly N_MELS = 80;
  private static readonly HOP_LENGTH = 160;
  private static readonly PREEMPHASIS = 0.97;
  private static readonly EPSILON = 1e-10;

  static async extractFeatures(samples: Float32Array): Promise<Float32Array> {
    let processedSamples = this.applyPreEmphasis(samples);
    if (processedSamples.length < this.WIN_LENGTH) {
      const padded = new Float32Array(this.WIN_LENGTH);
      padded.set(processedSamples);
      processedSamples = padded;
    }

    const numFrames = Math.max(
      1,
      Math.floor((processedSamples.length - this.WIN_LENGTH) / this.HOP_LENGTH) + 1
    );
    const melFeatures = new Float32Array(numFrames * this.N_MELS);
    const melFilterbank = this.createMelFilterbank(this.N_MELS, this.N_FFT, this.SAMPLE_RATE);
    const window = this.createHammingWindow(this.WIN_LENGTH);

    for (let frameIndex = 0; frameIndex < numFrames; frameIndex += 1) {
      const frameStart = frameIndex * this.HOP_LENGTH;
      const frame = new Float32Array(this.N_FFT);

      for (let i = 0; i < this.WIN_LENGTH; i += 1) {
        frame[i] = (processedSamples[frameStart + i] ?? 0) * window[i];
      }

      const spectrum = this.powerSpectrum(frame);
      for (let melIndex = 0; melIndex < this.N_MELS; melIndex += 1) {
        let energy = 0;
        const filter = melFilterbank[melIndex];
        for (let binIndex = 0; binIndex < filter.length; binIndex += 1) {
          energy += spectrum[binIndex] * filter[binIndex];
        }
        melFeatures[frameIndex * this.N_MELS + melIndex] = Math.log(Math.max(energy, this.EPSILON));
      }
    }

    this.applyPerBinMeanNormalization(melFeatures, numFrames, this.N_MELS);
    return melFeatures;
  }

  private static applyPreEmphasis(samples: Float32Array): Float32Array {
    if (samples.length === 0) return samples;
    const result = new Float32Array(samples.length);
    result[0] = samples[0];
    for (let i = 1; i < samples.length; i += 1) {
      result[i] = samples[i] - this.PREEMPHASIS * samples[i - 1];
    }
    return result;
  }

  private static createHammingWindow(size: number): Float32Array {
    const window = new Float32Array(size);
    for (let i = 0; i < size; i += 1) {
      window[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / Math.max(1, size - 1));
    }
    return window;
  }

  private static applyPerBinMeanNormalization(features: Float32Array, frames: number, bins: number) {
    if (frames <= 1) return;
    for (let bin = 0; bin < bins; bin += 1) {
      let sum = 0;
      for (let frame = 0; frame < frames; frame += 1) {
        sum += features[frame * bins + bin];
      }
      const mean = sum / frames;
      for (let frame = 0; frame < frames; frame += 1) {
        features[frame * bins + bin] -= mean;
      }
    }
  }

  private static createMelFilterbank(nMels: number, nFft: number, sampleRate: number): number[][] {
    const minMel = this.hzToMel(0);
    const maxMel = this.hzToMel(sampleRate / 2);
    const melPoints = new Float32Array(nMels + 2);
    for (let i = 0; i < melPoints.length; i += 1) {
      melPoints[i] = minMel + (i * (maxMel - minMel)) / (nMels + 1);
    }

    const hzPoints = Array.from(melPoints, (mel) => this.melToHz(mel));
    const bins = hzPoints.map((hz) => Math.floor(((nFft + 1) * hz) / sampleRate));
    const filterbank: number[][] = Array.from(
      { length: nMels },
      () => new Array(Math.floor(nFft / 2) + 1).fill(0)
    );

    for (let mel = 0; mel < nMels; mel += 1) {
      const left = bins[mel];
      const center = bins[mel + 1];
      const right = bins[mel + 2];

      for (let i = left; i < center; i += 1) {
        if (center !== left && i >= 0 && i < filterbank[mel].length) {
          filterbank[mel][i] = (i - left) / (center - left);
        }
      }
      for (let i = center; i < right; i += 1) {
        if (right !== center && i >= 0 && i < filterbank[mel].length) {
          filterbank[mel][i] = (right - i) / (right - center);
        }
      }
    }

    return filterbank;
  }

  private static hzToMel(hz: number) {
    return 2595 * Math.log10(1 + hz / 700);
  }

  private static melToHz(mel: number) {
    return 700 * (Math.pow(10, mel / 2595) - 1);
  }

  private static powerSpectrum(frame: Float32Array): Float32Array {
    const spectrumSize = Math.floor(frame.length / 2) + 1;
    const power = new Float32Array(spectrumSize);
    const n = frame.length;

    for (let k = 0; k < spectrumSize; k += 1) {
      let real = 0;
      let imag = 0;
      for (let i = 0; i < n; i += 1) {
        const angle = (2 * Math.PI * k * i) / n;
        real += frame[i] * Math.cos(angle);
        imag -= frame[i] * Math.sin(angle);
      }
      power[k] = (real * real + imag * imag) / n;
    }

    return power;
  }
}
