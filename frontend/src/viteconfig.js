import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// ─────────────────────────────────────────────────────────────
// 🔑 PUT YOUR ANTHROPIC API KEY HERE (or use a .env file)
//    Get your key at: https://console.anthropic.com/
//
//    OPTION A – paste directly (not recommended for production):
//      const ANTHROPIC_API_KEY = "sk-ant-...your-key-here...";
//
//    OPTION B – use a .env file (recommended):
//      1. Create a file called  .env  in your project root
//      2. Add this line:  VITE_ANTHROPIC_KEY=sk-ant-...your-key...
//      3. The config below reads it automatically
// ─────────────────────────────────────────────────────────────
const ANTHROPIC_API_KEY = process.env.VITE_ANTHROPIC_KEY || "PASTE_YOUR_API_KEY_HERE";

export default defineConfig({
  plugins: [react()],

  server: {
    // Allows page refresh on /dashboard without 404
    historyApiFallback: true,

    proxy: {
      // Any request to /api/claude gets forwarded to Anthropic
      "/api/claude": {
        target: "https://api.anthropic.com",
        changeOrigin: true,
        rewrite: (path) => "/v1/messages",
        configure: (proxy) => {
          proxy.on("proxyReq", (proxyReq) => {
            // Inject the API key and required headers server-side
            proxyReq.setHeader("x-api-key", ANTHROPIC_API_KEY);
            proxyReq.setHeader("anthropic-version", "2023-06-01");
            proxyReq.setHeader("content-type", "application/json");
          });
        },
      },
    },
  },
});