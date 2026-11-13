import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The frontend talks to the FastAPI backend at /api. In development Vite
// proxies that path to the Python server, so the browser sees a single
// origin and no CORS preflight is involved. In production the built assets
// are served by any static host and VITE_API_BASE points at the API.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      '/api': {
        target: process.env.VITE_API_TARGET || 'http://127.0.0.1:8000',
        changeOrigin: true,
        // Model loading and evaluation can take a while on a cold start;
        // the default proxy timeout would abort them.
        timeout: 600000,
        proxyTimeout: 600000,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    chunkSizeWarningLimit: 1600, // plotly is large and is split out below
    rollupOptions: {
      output: {
        manualChunks: {
          plotly: ['plotly.js-dist-min'],
          react: ['react', 'react-dom', 'react-router-dom'],
        },
      },
    },
  },
})
