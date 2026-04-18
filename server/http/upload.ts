import express from 'express';
import multer from 'multer';
import path from 'path';
import { PathManager } from '../path_manager.js';

const MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024;

function sanitizeUploadFilename(originalName: string) {
  const utf8Name = Buffer.from(originalName, 'latin1').toString('utf8');
  const base = path.basename(utf8Name);
  const cleaned = base
    .replace(/[<>:"/\\|?*\x00-\x1F]/g, '_')
    .replace(/\s+/g, ' ')
    .trim();

  if (!cleaned || cleaned === '.' || cleaned === '..') {
    return `upload_${Date.now()}`;
  }

  return cleaned.slice(0, 180);
}

export function createUploadHandlers() {
  const uploadStorage = multer.diskStorage({
    destination: (req, file, cb) => {
      try {
        const projectId = req.params.id;
        const ext = path.extname(file.originalname).toLowerCase();
        const subDir = ['.srt', '.vtt', '.ass', '.ssa'].includes(ext) ? 'subtitles' : 'assets';
        const projectPath = PathManager.getProjectPath(projectId, { create: true });
        const dest = path.join(projectPath, subDir);
        cb(null, dest);
      } catch (error) {
        cb(error as Error, '');
      }
    },
    filename: (req, file, cb) => {
      try {
        cb(null, sanitizeUploadFilename(file.originalname));
      } catch (error) {
        cb(error as Error, '');
      }
    },
  });

  const upload = multer({
    storage: uploadStorage,
    limits: {
      fileSize: MAX_UPLOAD_BYTES,
      files: 1,
    },
    fileFilter: (req, file, cb) => {
      const ext = path.extname(file.originalname).toLowerCase();
      const allowed = [
        '.mp4', '.mkv', '.mov', '.avi', '.wmv', '.webm', '.m4v', '.flv', '.ts',
        '.mp3', '.wav', '.aac', '.m4a', '.flac',
        '.srt', '.vtt', '.ass', '.ssa',
      ];
      if (!allowed.includes(ext)) {
        return cb(new Error('Unsupported file type'));
      }
      cb(null, true);
    },
  });

  const uploadTextOnly = multer({
    storage: uploadStorage,
    limits: {
      fileSize: 50 * 1024 * 1024,
      files: 1,
    },
    fileFilter: (req, file, cb) => {
      const ext = path.extname(file.originalname).toLowerCase();
      const allowed = ['.txt', '.srt', '.vtt', '.ass', '.ssa', '.json'];
      if (!allowed.includes(ext)) {
        return cb(new Error('Only text file types are allowed.'));
      }
      cb(null, true);
    },
  });

  const uploadSingle: express.RequestHandler = (req, res, next) => {
    upload.single('file')(req, res, (err: any) => {
      if (!err) return next();
      if (err instanceof multer.MulterError && err.code === 'LIMIT_FILE_SIZE') {
        return res.status(400).json({ error: 'File too large' });
      }
      return res.status(400).json({ error: err.message || 'Upload failed' });
    });
  };

  const uploadTextSingle: express.RequestHandler = (req, res, next) => {
    uploadTextOnly.single('file')(req, res, (err: any) => {
      if (!err) return next();
      if (err instanceof multer.MulterError && err.code === 'LIMIT_FILE_SIZE') {
        return res.status(400).json({ error: 'File too large' });
      }
      return res.status(400).json({ error: err.message || 'Upload failed' });
    });
  };

  return { uploadSingle, uploadTextSingle };
}
