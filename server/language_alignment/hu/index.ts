import { createSimpleWhitespaceForcedAlignmentModule } from '../shared/simple_whitespace_forced_alignment.js';
import type { LanguageAlignmentPack } from '../shared/types.js';

export const languagePack: LanguageAlignmentPack = {
  key: 'hu',
  aliases: ['hu', 'hu-hu'],
  forcedAlignment: createSimpleWhitespaceForcedAlignmentModule(import.meta.url),
};

export const hungarianLanguagePack = languagePack;
