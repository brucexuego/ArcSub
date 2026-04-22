import { createSimpleWhitespaceForcedAlignmentModule } from '../shared/simple_whitespace_forced_alignment.js';
import type { LanguageAlignmentPack } from '../shared/types.js';

export const languagePack: LanguageAlignmentPack = {
  key: 'fa',
  aliases: ['fa', 'fa-ir', 'fa-af'],
  forcedAlignment: createSimpleWhitespaceForcedAlignmentModule(import.meta.url),
};

export const persianLanguagePack = languagePack;
