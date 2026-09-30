import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Cloud-sync folders/antivirus can briefly lock hashed assets from the previous build.
  // Vite still replaces index.html and writes the new hashes; keeping old ignored
  // artifacts makes repeat local builds reliable without deleting user files.
  build: { emptyOutDir: false },
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8001',
        changeOrigin: false,
      },
    },
  },
})
