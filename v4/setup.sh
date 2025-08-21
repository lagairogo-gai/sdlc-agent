# scripts/setup.sh - Setup Script
#!/bin/bash

echo "🚀 Setting up RAG User Stories Generator..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p backend/uploads
mkdir -p backend/logs
mkdir -p nginx/ssl
mkdir -p backup

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your actual API keys and configuration"
fi

# Generate SSL certificates for development
if [ ! -f nginx/ssl/cert.pem ]; then
    echo "🔒 Generating SSL certificates for development..."
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout nginx/ssl/key.pem \
        -out nginx/ssl/cert.pem \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
fi

# Set correct permissions
echo "🔧 Setting permissions..."
chmod +x scripts/*.sh
chmod 600 nginx/ssl/*.pem

# Build and start services
echo "🏗️  Building and starting services..."
docker-compose build
docker-compose up -d

echo "⏳ Waiting for services to start..."
sleep 30

# Check if services are running
echo "🔍 Checking service health..."
if curl -f http://localhost:3001/health > /dev/null 2>&1; then
    echo "✅ Backend is running"
else
    echo "❌ Backend is not responding"
fi

if curl -f http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ Frontend is running"
else
    echo "❌ Frontend is not responding"
fi

echo ""
echo "🎉 Setup complete!"
echo "Frontend: http://localhost:3000"
echo "Backend API: http://localhost:3001"
echo "MongoDB: mongodb://localhost:27017"
echo "Redis: redis://localhost:6379"
echo "Weaviate: http://localhost:8080"
echo ""
echo "📝 Don't forget to:"
echo "1. Edit .env with your API keys"
echo "2. Configure your data source integrations"
echo "3. Check logs with: docker-compose logs -f"