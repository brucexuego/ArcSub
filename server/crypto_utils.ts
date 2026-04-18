import crypto from 'crypto';
import dotenv from 'dotenv';

dotenv.config();

const ENCRYPTION_KEY = process.env.ENCRYPTION_KEY;

if (!ENCRYPTION_KEY) {
  console.warn('\x1b[33m%s\x1b[0m', ' [SECURITY WARNING] ENCRYPTION_KEY not found in .env! Using default insecure key.');
  console.warn('\x1b[33m%s\x1b[0m', ' Please set ENCRYPTION_KEY in your .env file to properly secure your API keys.');
}

const FINAL_KEY = ENCRYPTION_KEY || 'default-very-secret-key-32-chars!!';
const IV_LENGTH = 16; 

const derivatedKey = crypto.createHash('sha256').update(FINAL_KEY).digest();

export function encrypt(text: string): string {
  if (!text) return '';
  const iv = crypto.randomBytes(IV_LENGTH);
  const cipher = crypto.createCipheriv('aes-256-cbc', derivatedKey, iv);
  let encrypted = cipher.update(text);
  encrypted = Buffer.concat([encrypted, cipher.final()]);
  return iv.toString('hex') + ':' + encrypted.toString('hex');
}

export function decrypt(text: string): string {
  if (!text) return '';
  try {
    const textParts = text.split(':');
    const iv = Buffer.from(textParts.shift()!, 'hex');
    const encryptedText = Buffer.from(textParts.join(':'), 'hex');
    const decipher = crypto.createDecipheriv('aes-256-cbc', derivatedKey, iv);
    let decrypted = decipher.update(encryptedText);
    decrypted = Buffer.concat([decrypted, decipher.final()]);
    return decrypted.toString();
  } catch (e) {
    console.error('Decryption failed. Data might be plain text or corrupted.', e);
    return text; // Fallback to original text if decryption fails (e.g. migrate from plain text)
  }
}
