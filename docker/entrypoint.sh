#!/bin/bash
set -e

# SwellSight Inference Service Entrypoint Script
# Handles initialization, health checks, and graceful startup

echo "Starting SwellSight Inference Service..."

# Set default values for environment variables
export MODEL_REGISTRY_PATH=${MODEL_REGISTRY_PATH:-"/app/models/registry"}
export API_HOST=${API_HOST:-"0.0.0.0"}
export API_PORT=${API_PORT:-"8000"}
export LOG_LEVEL=${LOG_LEVEL:-"INFO"}
export WORKERS=${WORKERS:-"1"}
export ENABLE_AUTH=${ENABLE_AUTH:-"false"}

# Create necessary directories
mkdir -p "$MODEL_REGISTRY_PATH"
mkdir -p /app/logs

# Check if model registry exists and has models
if [ ! -f "$MODEL_REGISTRY_PATH/registry.json" ]; then
    echo "Warning: No model registry found at $MODEL_REGISTRY_PATH"
    echo "The service will start but may not be able to serve predictions until models are loaded."
fi

# Health check function
health_check() {
    local max_attempts=30
    local attempt=1
    
    echo "Waiting for service to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
            echo "Service is ready!"
            return 0
        fi
        
        echo "Attempt $attempt/$max_attempts: Service not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "Service failed to become ready within expected time"
    return 1
}

# Signal handlers for graceful shutdown
cleanup() {
    echo "Received shutdown signal, stopping service..."
    if [ ! -z "$API_PID" ]; then
        kill -TERM "$API_PID" 2>/dev/null || true
        wait "$API_PID" 2>/dev/null || true
    fi
    echo "Service stopped gracefully"
    exit 0
}

trap cleanup SIGTERM SIGINT

# Start the API service
echo "Starting API service on $API_HOST:$API_PORT with $WORKERS workers..."

# Build command arguments
CMD_ARGS=(
    "--host" "$API_HOST"
    "--port" "$API_PORT"
    "--registry-path" "$MODEL_REGISTRY_PATH"
    "--workers" "$WORKERS"
)

# Add authentication if enabled
if [ "$ENABLE_AUTH" = "true" ]; then
    CMD_ARGS+=("--enable-auth")
fi

# Add CORS origins if specified
if [ ! -z "$CORS_ORIGINS" ]; then
    IFS=',' read -ra ORIGINS <<< "$CORS_ORIGINS"
    for origin in "${ORIGINS[@]}"; do
        CMD_ARGS+=("--cors-origins" "$origin")
    done
fi

# Start the service in background for health checking
python -m swellsight.api.production_api "${CMD_ARGS[@]}" &
API_PID=$!

# Wait a moment for the service to start
sleep 5

# Perform health check if requested
if [ "$SKIP_HEALTH_CHECK" != "true" ]; then
    if ! health_check; then
        echo "Health check failed, stopping service"
        cleanup
        exit 1
    fi
fi

# Wait for the API process
wait "$API_PID"