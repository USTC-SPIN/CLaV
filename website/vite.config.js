import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Base path matches the GitHub Pages URL: https://patience-joey.github.io/clav/
export default defineConfig({
  base: '/clav/',
  plugins: [react()],
  build: {
    outDir: 'dist',
    assetsInlineLimit: 2048,
  },
})
