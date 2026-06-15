# SvelteKit frontend image (SP-4 detect UI + SP-5 dashboard/history).
#
# Two stages: (1) install + build with adapter-node into /app/build; (2) a slim runtime that
# just runs `node build`. Served behind nginx-proxy via docker-compose VIRTUAL_HOST=
# htr.eginugraha.com on port 3000.
#
# IMPORTANT: the browser calls the backend DIRECTLY (client-side fetch), and Vite INLINES
# VITE_API_BASE at build time (import.meta.env). So the backend URL must be passed as a build
# ARG here — it cannot be changed at container runtime.

# ---- Stage 1: build ----------------------------------------------------------------------
FROM node:22-alpine AS build
WORKDIR /app

# Lockfile-first install so this layer caches unless dependencies change.
COPY package.json package-lock.json ./
RUN npm ci

# App source (node_modules / build / .svelte-kit excluded via .dockerignore).
COPY . .

# Public URL of the backend API. Baked into the client bundle at build time.
ARG VITE_API_BASE=https://htr-api.eginugraha.com
ENV VITE_API_BASE=$VITE_API_BASE

# Produce build/ (adapter-node) and drop dev dependencies so we copy a slim node_modules.
RUN npm run build && npm prune --omit=dev

# ---- Stage 2: runtime --------------------------------------------------------------------
FROM node:22-alpine
WORKDIR /app

# adapter-node reads HOST/PORT from the environment; 0.0.0.0:3000 makes it reachable by the proxy.
ENV NODE_ENV=production \
    HOST=0.0.0.0 \
    PORT=3000

# Copy only what `node build` needs: the built server, production deps, and package.json.
COPY --from=build /app/build ./build
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/package.json ./package.json

EXPOSE 3000
CMD ["node", "build"]
