# Route Similarity Backend (Kotlin/Spring Boot)

Lightweight route similarity service for mobile app backend.

## Memory Usage (~500MB for 10k routes)

| Component | Memory |
|-----------|--------|
| JVM base | ~300 MB |
| Spring Boot | ~150 MB |
| 10,000 embeddings (128-dim) | ~5 MB |
| **Total** | **~455 MB** |

✅ **Fits easily in 2GB Lightsail!**

## Setup

### 1. Add Dependencies (build.gradle.kts)

```kotlin
dependencies {
    implementation("org.springframework.boot:spring-boot-starter-web")
    implementation("com.fasterxml.jackson.module:jackson-module-kotlin")
}
```

### 2. Copy Export Files

From your Python notebook, run the export cell, then:

```bash
# Copy exported files to Spring Boot resources
mkdir -p src/main/resources/data
cp export_kotlin/embeddings.bin src/main/resources/data/
cp export_kotlin/routes.json src/main/resources/data/
```

### 3. Add Service Files

Copy these Kotlin files to your Spring Boot project:
- `RouteSimilarityService.kt` → `src/main/kotlin/com/yourapp/service/`
- `RouteController.kt` → `src/main/kotlin/com/yourapp/controller/`

### 4. Update Package Names

Replace `com.yourapp` with your actual package name.

## API Endpoints

### Get Similar Routes
```
GET /api/routes/{routeId}/similar?topK=10
```

**Response:**
```json
{
  "success": true,
  "queryRouteId": "2024-01-15T08-30-00.gpx",
  "queryRoute": {
    "id": "2024-01-15T08-30-00.gpx",
    "terrain": "road",
    "distance_km": 5.2,
    "pace_min_per_km": 5.5,
    "runner_type": "average"
  },
  "similarRoutes": [
    {
      "routeId": "2024-01-10T07-15-00.gpx",
      "similarityPercent": 92.5,
      "terrain": "road",
      "distanceKm": 5.0,
      "paceMinPerKm": 5.3,
      "runnerType": "average"
    }
  ],
  "totalIndexedRoutes": 1000
}
```

### Get Route Info
```
GET /api/routes/{routeId}
```

### Get Service Stats
```
GET /api/routes/stats
```

## Deployment on AWS Lightsail

### application.properties
```properties
# Use external directory for embeddings (easier updates)
EMBEDDINGS_DIR=/app/data

# Or use classpath (packaged in JAR)
# EMBEDDINGS_DIR=classpath:data
```

### Run with Docker
```dockerfile
FROM eclipse-temurin:21-jre-alpine
WORKDIR /app
COPY target/*.jar app.jar
COPY export_kotlin/ /app/data/
ENV EMBEDDINGS_DIR=/app/data
EXPOSE 8080
ENTRYPOINT ["java", "-Xmx512m", "-jar", "app.jar"]
```

### Run with systemd
```bash
# Copy files
scp target/*.jar lightsail:/app/
scp -r export_kotlin/ lightsail:/app/data/

# Create service file
sudo systemctl enable myapp
sudo systemctl start myapp
```

## Updating Embeddings

When you add new routes to your dataset:

1. Re-run the export cell in the notebook
2. Copy new `embeddings.bin` and `routes.json` to server
3. Restart the service (or call reload endpoint if implemented)

No code changes needed!

## Performance

- **Similarity search**: ~1ms for 10k routes (cosine similarity)
- **Memory**: ~5KB per 1000 routes (128-dim float32)
- **Startup**: ~2s to load embeddings

## Future Enhancements

For 100k+ routes, consider:
- FAISS-lite or Annoy for approximate nearest neighbors
- Redis for embedding storage
- Batch processing for new routes
