# frontend/Dockerfile

FROM node:20-alpine AS build

WORKDIR /app

COPY package*.json ./
RUN npm ci 

COPY . .

RUN cp .env.docker .env.local

RUN npm run build

FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["/bin/sh", "-c", "echo 'Frontend running on http://localhost:3000 (mapped via docker-compose)'; nginx -g 'daemon off;'"]