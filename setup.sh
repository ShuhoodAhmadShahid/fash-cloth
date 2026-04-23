#!/bin/bash

# AI Fashion Recommendation System - Setup Script
# This script helps you set up and run the application

set -e

echo "=========================================="
echo "AI Fashion Recommendation System Setup"
echo "=========================================="
echo ""

# Check if Docker is available
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo "✓ Docker and Docker Compose found!"
    echo ""
    echo "Starting with Docker (Recommended)..."
    echo ""
    
    cd docker
    docker-compose up --build
    
else
    echo "Docker not found. Setting up for local installation..."
    echo ""
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo "✗ Python not found! Please install Python 3.8+"
        exit 1
    fi
    
    echo "✓ Using Python: $PYTHON_CMD"
    echo ""
    
    # Create virtual environment (optional but recommended)
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    source venv/bin/activate
    
    echo "Installing dependencies..."
    cd backend
    pip install --upgrade pip
    pip install -r requirements.txt
    
    echo ""
    echo "=========================================="
    echo "Setup Complete!"
    echo "=========================================="
    echo ""
    echo "To start the application:"
    echo "  cd backend"
    echo "  source ../venv/bin/activate  # If using venv"
    echo "  python app.py"
    echo ""
    echo "Then open http://localhost:5000 in your browser"
    echo ""
fi
