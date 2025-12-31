#!/bin/bash

# SwellSight Inference Service Deployment Script
# Supports local Docker, Docker Compose, and Kubernetes deployments

set -e

# Default values
DEPLOYMENT_TYPE="docker-compose"
ENVIRONMENT="development"
IMAGE_TAG="latest"
NAMESPACE="swellsight"
REGISTRY="ghcr.io/swellsight"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
SwellSight Inference Service Deployment Script

Usage: $0 [OPTIONS]

OPTIONS:
    -t, --type TYPE         Deployment type: docker, docker-compose, k8s (default: docker-compose)
    -e, --env ENV          Environment: development, staging, production (default: development)
    -i, --image TAG        Docker image tag (default: latest)
    -n, --namespace NS     Kubernetes namespace (default: swellsight)
    -r, --registry REG     Docker registry (default: ghcr.io/swellsight)
    -b, --build           Build image locally before deployment
    -c, --clean           Clean up existing deployment before deploying
    -h, --help            Show this help message

EXAMPLES:
    # Deploy with Docker Compose (default)
    $0

    # Deploy to Kubernetes with custom image
    $0 --type k8s --image v1.2.3 --env production

    # Build and deploy locally
    $0 --type docker --build --clean

    # Deploy to staging environment
    $0 --type k8s --env staging --namespace swellsight-staging

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            DEPLOYMENT_TYPE="$2"
            shift 2
            ;;
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -i|--image)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -b|--build)
            BUILD_IMAGE=true
            shift
            ;;
        -c|--clean)
            CLEAN_DEPLOYMENT=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate deployment type
case $DEPLOYMENT_TYPE in
    docker|docker-compose|k8s)
        ;;
    *)
        log_error "Invalid deployment type: $DEPLOYMENT_TYPE"
        log_error "Valid types: docker, docker-compose, k8s"
        exit 1
        ;;
esac

# Set image name
IMAGE_NAME="$REGISTRY/inference:$IMAGE_TAG"

log_info "Starting SwellSight Inference Service deployment"
log_info "Deployment type: $DEPLOYMENT_TYPE"
log_info "Environment: $ENVIRONMENT"
log_info "Image: $IMAGE_NAME"

# Build image if requested
if [[ "$BUILD_IMAGE" == "true" ]]; then
    log_info "Building Docker image..."
    docker build -f docker/Dockerfile.inference -t "$IMAGE_NAME" .
    log_success "Image built successfully"
fi

# Clean up existing deployment if requested
if [[ "$CLEAN_DEPLOYMENT" == "true" ]]; then
    log_info "Cleaning up existing deployment..."
    
    case $DEPLOYMENT_TYPE in
        docker)
            docker stop swellsight-inference 2>/dev/null || true
            docker rm swellsight-inference 2>/dev/null || true
            ;;
        docker-compose)
            docker-compose -f docker-compose.inference.yml down
            ;;
        k8s)
            kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
            ;;
    esac
    
    log_success "Cleanup completed"
fi

# Deploy based on type
case $DEPLOYMENT_TYPE in
    docker)
        deploy_docker
        ;;
    docker-compose)
        deploy_docker_compose
        ;;
    k8s)
        deploy_kubernetes
        ;;
esac

# Deployment functions
deploy_docker() {
    log_info "Deploying with Docker..."
    
    # Create necessary directories
    mkdir -p ./models ./logs
    
    # Run container
    docker run -d \
        --name swellsight-inference \
        -p 8000:8000 \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/logs:/app/logs" \
        -e MODEL_REGISTRY_PATH=/app/models/registry \
        -e API_HOST=0.0.0.0 \
        -e API_PORT=8000 \
        -e LOG_LEVEL=INFO \
        -e WORKERS=2 \
        -e ENABLE_AUTH=false \
        "$IMAGE_NAME"
    
    log_success "Docker deployment completed"
    log_info "Service available at: http://localhost:8000"
    log_info "Health check: http://localhost:8000/health"
    log_info "API docs: http://localhost:8000/docs"
}

deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    # Set environment variables for docker-compose
    export SWELLSIGHT_IMAGE="$IMAGE_NAME"
    export SWELLSIGHT_ENV="$ENVIRONMENT"
    
    # Deploy based on environment
    if [[ "$ENVIRONMENT" == "production" ]]; then
        docker-compose -f docker-compose.inference.yml --profile monitoring --profile cache up -d
    else
        docker-compose -f docker-compose.inference.yml up -d
    fi
    
    log_success "Docker Compose deployment completed"
    log_info "Service available at: http://localhost:8000"
    log_info "Health check: http://localhost:8000/health"
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log_info "Monitoring available at: http://localhost:9090 (Prometheus)"
        log_info "Dashboards available at: http://localhost:3000 (Grafana)"
    fi
}

deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Create namespace
    kubectl apply -f k8s/namespace.yaml
    
    # Update image in deployment manifest
    sed -i.bak "s|image: swellsight/inference:.*|image: $IMAGE_NAME|" k8s/deployment.yaml
    
    # Update namespace in manifests if different from default
    if [[ "$NAMESPACE" != "swellsight" ]]; then
        for file in k8s/*.yaml; do
            sed -i.bak "s|namespace: swellsight|namespace: $NAMESPACE|g" "$file"
        done
    fi
    
    # Apply manifests
    log_info "Applying Kubernetes manifests..."
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/pvc.yaml
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml
    
    # Apply HPA for production
    if [[ "$ENVIRONMENT" == "production" ]]; then
        kubectl apply -f k8s/hpa.yaml
    fi
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl rollout status deployment/swellsight-inference -n "$NAMESPACE" --timeout=300s
    
    # Get service information
    SERVICE_TYPE=$(kubectl get service swellsight-inference-service -n "$NAMESPACE" -o jsonpath='{.spec.type}')
    
    if [[ "$SERVICE_TYPE" == "LoadBalancer" ]]; then
        log_info "Waiting for LoadBalancer to be ready..."
        kubectl wait --for=jsonpath='{.status.loadBalancer.ingress}' service/swellsight-inference-service -n "$NAMESPACE" --timeout=300s
        
        EXTERNAL_IP=$(kubectl get service swellsight-inference-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        if [[ -z "$EXTERNAL_IP" ]]; then
            EXTERNAL_IP=$(kubectl get service swellsight-inference-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
        fi
        
        log_success "Kubernetes deployment completed"
        log_info "Service available at: http://$EXTERNAL_IP"
        log_info "Health check: http://$EXTERNAL_IP/health"
    else
        CLUSTER_IP=$(kubectl get service swellsight-inference-service -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
        log_success "Kubernetes deployment completed"
        log_info "Service available internally at: http://$CLUSTER_IP:80"
        log_info "Use 'kubectl port-forward' to access from outside the cluster"
    fi
    
    # Restore original manifests
    for file in k8s/*.yaml.bak; do
        if [[ -f "$file" ]]; then
            mv "$file" "${file%.bak}"
        fi
    done
}

# Health check function
check_health() {
    local url="$1"
    local max_attempts=30
    local attempt=1
    
    log_info "Performing health check..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s "$url/health" > /dev/null 2>&1; then
            log_success "Health check passed"
            return 0
        fi
        
        log_info "Health check attempt $attempt/$max_attempts..."
        sleep 5
        attempt=$((attempt + 1))
    done
    
    log_error "Health check failed after $max_attempts attempts"
    return 1
}

# Perform health check based on deployment type
case $DEPLOYMENT_TYPE in
    docker|docker-compose)
        check_health "http://localhost:8000"
        ;;
    k8s)
        if [[ -n "$EXTERNAL_IP" ]]; then
            check_health "http://$EXTERNAL_IP"
        else
            log_info "Skipping external health check for ClusterIP service"
        fi
        ;;
esac

log_success "Deployment completed successfully!"

# Show useful commands
log_info "Useful commands:"
case $DEPLOYMENT_TYPE in
    docker)
        echo "  View logs: docker logs swellsight-inference"
        echo "  Stop service: docker stop swellsight-inference"
        echo "  Remove container: docker rm swellsight-inference"
        ;;
    docker-compose)
        echo "  View logs: docker-compose -f docker-compose.inference.yml logs -f"
        echo "  Stop services: docker-compose -f docker-compose.inference.yml down"
        echo "  Scale service: docker-compose -f docker-compose.inference.yml up -d --scale swellsight-inference=3"
        ;;
    k8s)
        echo "  View logs: kubectl logs -f deployment/swellsight-inference -n $NAMESPACE"
        echo "  Scale deployment: kubectl scale deployment swellsight-inference --replicas=5 -n $NAMESPACE"
        echo "  Port forward: kubectl port-forward service/swellsight-inference-service 8000:80 -n $NAMESPACE"
        echo "  Delete deployment: kubectl delete namespace $NAMESPACE"
        ;;
esac