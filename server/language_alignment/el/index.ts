import { createSimpleWhitespaceForcedAlignmentModule } from '../shared/simple_whitespace_forced_alignment.js';
import type { LanguageAlignmentPack } from '../shared/types.js';

export const languagePack: LanguageAlignmentPack = {
  key: 'el',
  aliases: ['el', 'el-gr'],
  forcedAlignment: createSimpleWhitespaceForcedAlignmentModule(import.meta.url),
};

export const greekLanguagePack = languagePack;
