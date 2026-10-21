import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

// The base path is injected by CI (see .github/workflows/pages.yml) so the same
// code deploys to any <owner>.github.io/<repo>/ or a custom domain.
function resolveBase(): string {
  const raw = process.env.VITE_BASE ?? '/';
  const trimmed = raw.replace(/^\/+|\/+$/g, '');
  return trimmed ? `/${trimmed}/` : '/';
}

export default defineConfig({
  base: resolveBase(),
  plugins: [react()],
  worker: { format: 'es' },
  build: { target: 'es2022', chunkSizeWarningLimit: 4000 },
  test: {
    include: ['tests/**/*.test.ts'],
    environment: 'node',
  },
});
