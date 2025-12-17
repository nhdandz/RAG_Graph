#!/bin/bash

echo "Starting RAG Demo Backend..."
echo "=============================="

# Check if Java is installed
if ! command -v java &> /dev/null; then
    echo "Error: Java is not installed. Please install Java 17 or higher."
    exit 1
fi

# Check Java version
JAVA_VERSION=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}' | cut -d'.' -f1)
if [ "$JAVA_VERSION" -lt 17 ]; then
    echo "Error: Java 17 or higher is required. Current version: $JAVA_VERSION"
    exit 1
fi

# Check if Maven is installed
if ! command -v mvn &> /dev/null; then
    echo "Error: Maven is not installed. Please install Maven 3.6 or higher."
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
    echo "Warning: Ollama does not appear to be running on port 11434."
    echo "Please start Ollama with: ollama serve"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

cd backend

echo "Building project..."
mvn clean install -DskipTests

if [ $? -eq 0 ]; then
    echo ""
    echo "Build successful! Starting application..."
    echo "Backend will be available at: http://localhost:8080"
    echo "Press Ctrl+C to stop"
    echo ""
    mvn spring-boot:run
else
    echo "Build failed. Please check the errors above."
    exit 1
fi
