#!/bin/bash

# Run any setups or pre-processing tasks here
echo "Starting hospital RAG FastAPI service..."

# Start the main application
uvicorn main:app --host 0.0.0.0 --port 8000
