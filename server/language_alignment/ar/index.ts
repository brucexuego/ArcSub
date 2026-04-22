import { createSimpleWhitespaceForcedAlignmentModule } from '../shared/simple_whitespace_forced_alignment.js';
import type { LanguageAlignmentPack } from '../shared/types.js';

export const languagePack: LanguageAlignmentPack = {
  key: 'ar',
  aliases: ['ar', 'ar-eg', 'ar-sa', 'ar-ae', 'ar-jo', 'ar-ly', 'ar-tn', 'ar-ma'],
  forcedAlignment: createSimpleWhitespaceForcedAlignmentModule(import.meta.url),
};

export const arabicLanguagePack = languagePack;

