#!/bin/bash
echo "============================================"
echo " Weather AI Advisor - Docker Build & Run"
echo "============================================"
echo ""
echo "Building Docker image..."
docker build -t weather-ai-advisor .
echo ""
echo "Starting container on port 5000..."
docker run -d -p 5000:5000 --name weather-advisor weather-ai-advisor
echo ""
echo "Container started! API available at:"
echo "  http://localhost:5000"
echo ""
echo "Test commands:"
echo "  curl -X POST http://localhost:5000/predict -H \"Content-Type: application/json\" -d @test_input.json"