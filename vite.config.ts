import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig, loadEnv} from 'vite';

function parseAllowedHosts(value: string | undefined) {
  return String(value || '')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const allowedHosts = parseAllowedHosts(env.ARCSUB_DEV_ALLOWED_HOSTS);

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
      ...(allowedHosts.length > 0 ? { allowedHosts } : {}),
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      hmr: process.env.DISABLE_HMR !== 'true',
      watch: {
        ignored: ['**/db.json', '**/server/**', '**/Projects/**', '**/tmp/**', '**/local/**', '**/__pycache__/**'],
      },
    },
  };
});
