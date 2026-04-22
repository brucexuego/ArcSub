import { createSimpleWhitespaceForcedAlignmentModule } from '../shared/simple_whitespace_forced_alignment.js';
import type { LanguageAlignmentPack } from '../shared/types.js';

export const languagePack: LanguageAlignmentPack = {
  key: 'nl',
  aliases: ['nl', 'nl-nl', 'nl-be'],
  forcedAlignment: createSimpleWhitespaceForcedAlignmentModule(import.meta.url),
};

export const dutchLanguagePack = languagePack;
