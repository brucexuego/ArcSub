import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);

type OpenCcModule = {
  Converter(options: { from: string; to: string }): (input: string) => string;
};

export class ChineseScriptNormalizer {
  private static openCcModule: OpenCcModule | null = null;
  private static taiwanTraditionalConverter: ((input: string) => string) | null = null;
  private static openCcUnavailable = false;

  private static getOpenCcModule() {
    if (this.openCcUnavailable) return null;
    if (this.openCcModule) return this.openCcModule;

    try {
      this.openCcModule = require('opencc-js') as OpenCcModule;
      return this.openCcModule;
    } catch (error) {
      this.openCcUnavailable = true;
      console.warn('[translation] OpenCC normalization unavailable; falling back to raw zh-TW output.', error);
      return null;
    }
  }

  private static getTaiwanTraditionalConverter() {
    const OpenCC = this.getOpenCcModule();
    if (!OpenCC) return null;

    if (!this.taiwanTraditionalConverter) {
      this.taiwanTraditionalConverter = OpenCC.Converter({ from: 'cn', to: 'twp' });
    }
    return this.taiwanTraditionalConverter;
  }

  static normalizeToTaiwanTraditional(text: string) {
    const source = String(text || '');
    if (!source || this.openCcUnavailable) return source;

    try {
      const converter = this.getTaiwanTraditionalConverter();
      if (!converter) return source;
      return converter(source);
    } catch (error) {
      this.openCcUnavailable = true;
      console.warn('[translation] OpenCC normalization unavailable; falling back to raw zh-TW output.', error);
      return source;
    }
  }
}
