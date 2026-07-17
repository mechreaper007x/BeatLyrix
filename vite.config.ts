import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './'),
    },
  },
  server: {
    port: 3000,
    watch: {
      ignored: [
        '**/raprank-backend/**',
        '**/raprank-nlp/**',
        '**/raprank-semantic/**',
        '**/raprank-upload/**',
        '**/local_real_model/**',
        '**/kaggle_rap_data/**',
      ],
    },
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      '/uploads': {
        target: 'http://localhost:9090',
        changeOrigin: true,
      },
    },
  },
});
