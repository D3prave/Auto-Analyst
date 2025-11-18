# frontend/Dockerfile

# ---------- Build stage ----------
FROM node:20-alpine AS build

WORKDIR /app

# Install deps
COPY package*.json ./
RUN npm install

# Copy source
COPY . .

# Use Docker-specific env so Vite bakes the correct API URL
# (we copy .env.docker to .env.local so Vite sees it)
RUN cp .env.docker .env.local

# Build static files
RUN npm run build

# ---------- Runtime stage ----------
FROM nginx:alpine

# Copy built frontend to Nginx's html directory
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

# UPDATED: Changed 0.0.0.0 to localhost for clarity
CMD ["/bin/sh", "-c", "echo 'Frontend running on http://localhost:3000 (mapped via docker-compose)'; nginx -g 'daemon off;'"]