import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { localAnalysisPlugin } from "./server/local-analysis.js";


const repoRoot = path.resolve(__dirname, "..");

export default defineConfig({
  plugins: [react(), localAnalysisPlugin({ repoRoot })],
  server: {
    port: 4173,
    strictPort: false,
    fs: {
      allow: [repoRoot],
    },
  },
  preview: {
    port: 4173,
    strictPort: false,
  },
});
