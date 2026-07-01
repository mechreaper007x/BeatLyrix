# ─────────────────────────────────────────────────────────────────────────────
# RapRank — Go Audio Upload Service
# Multi-stage: Go build → minimal Alpine runtime
# ─────────────────────────────────────────────────────────────────────────────

# Stage 1: compile
FROM golang:1.22-alpine AS build
WORKDIR /app

COPY go.mod .
# go.sum may not exist if there are no external deps — touch it to be safe
RUN touch go.sum && go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o raprank-upload .

# Stage 2: minimal runtime
FROM alpine:3.19
WORKDIR /app

# curl for healthcheck inside compose
RUN apk add --no-cache curl

# Upload directory (mounted as a volume in docker-compose)
RUN mkdir -p uploads/audio

COPY --from=build /app/raprank-upload .

EXPOSE 9090

CMD ["./raprank-upload"]
