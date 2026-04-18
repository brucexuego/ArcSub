import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig} from 'vite';

export default defineConfig(() => {
  return {
    plugins: [react(), tailwindcss()],
    build: {
      rollupOptions: {
        output: {
          manualChunks(id) {
            const normalizedId = id.replace(/\\/g, '/');
            if (!normalizedId.includes('/node_modules/')) return;

            if (
              normalizedId.includes('/node_modules/react/') ||
              normalizedId.includes('/node_modules/react-dom/') ||
              normalizedId.includes('/node_modules/scheduler/')
            ) {
              return 'react-vendor';
            }

            if (normalizedId.includes('/node_modules/lucide-react/')) {
              return 'icon-vendor';
            }

            if (normalizedId.includes('/node_modules/artplayer/')) {
              return 'player-vendor';
            }

            if (normalizedId.includes('/node_modules/motion/')) {
              return 'motion-vendor';
            }

            if (normalizedId.includes('/node_modules/@huggingface/transformers/')) {
              return 'hf-vendor';
            }

            return 'vendor';
          },
        },
      },
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      hmr: process.env.DISABLE_HMR !== 'true',
      watch: {
        ignored: ['**/db.json', '**/server/**', '**/Projects/**', '**/tmp/**', '**/local/**', '**/__pycache__/**'],
      },
    },
  };
});
