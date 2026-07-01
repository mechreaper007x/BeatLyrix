# ─────────────────────────────────────────────────────────────────────────────
# RapRank — Spring Boot Backend
# Multi-stage: Maven build → minimal JRE runtime
# ─────────────────────────────────────────────────────────────────────────────

# Stage 1: build the fat JAR
FROM maven:3.9-eclipse-temurin-21-alpine AS build
WORKDIR /app

# Copy pom first for layer caching — dependencies won't re-download unless pom changes
COPY pom.xml .
RUN mvn dependency:go-offline -q

# Copy source and build (skip tests — they need a live DB)
COPY src ./src
RUN mvn clean package -DskipTests -q

# Stage 2: minimal JRE runtime
FROM eclipse-temurin:21-jre-alpine
WORKDIR /app

# Upload directory (shared with raprank-upload via Docker volume)
RUN mkdir -p uploads/audio

COPY --from=build /app/target/*.jar app.jar

EXPOSE 8080

# All secrets are injected via docker-compose environment variables.
# Spring Boot maps SPRING_DATASOURCE_URL → spring.datasource.url etc.
ENTRYPOINT ["java", "-jar", "app.jar"]
