import { defineConfig } from "vite";
import path from "node:path";
import { fileURLToPath } from "node:url";

const frontendRoot = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  build: {
    target: "es2020",
    rollupOptions: {
      input: {
        model: path.resolve(frontendRoot, "index.html"),
        stats: path.resolve(frontendRoot, "stats.html"),
        usage: path.resolve(frontendRoot, "usage.html"),
        api: path.resolve(frontendRoot, "api.html"),
        test: path.resolve(frontendRoot, "test.html")
      }
    }
  },
  server: {
    fs: {
      allow: [".."]
    }
  }
});
