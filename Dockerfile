FROM node:23

WORKDIR /app

COPY package.json pnpm-lock.yaml ./
RUN corepack enable pnpm && pnpm i --frozen-lockfile

COPY . ./

CMD ["pnpm", "start:server"]
