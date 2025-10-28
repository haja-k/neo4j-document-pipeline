#!/bin/bash
set -e  # Exit immediately if any command fails

echo "🔧 Building Docker images..."

# Build the Neo4j base image (from official source)
echo "➡️  Pulling latest Neo4j image..."
docker pull neo4j:5

# Build your API image
echo "➡️  Building neo4j-api:v0.1 image..."
docker build -t neo4j-api:v0.1 .

echo "✅ Build complete!"
echo "You can now run: docker-compose up -d --build"
