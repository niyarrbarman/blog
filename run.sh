#!/bin/bash

# Elegant Academic Blog - Jekyll Server Script
# This script sets up and runs the Jekyll development server

set -e

echo "🚀 Starting Elegant Academic Blog..."
echo ""

# Check if Ruby is installed
if ! command -v ruby &> /dev/null; then
    echo "❌ Ruby is not installed. Please install Ruby first."
    echo "   On Ubuntu/Debian: sudo apt install ruby-full"
    echo "   On macOS: brew install ruby"
    exit 1
fi

# Check if Bundler is installed
if ! command -v bundle &> /dev/null; then
    echo "📦 Installing Bundler..."
    gem install bundler
fi

# Install dependencies
if [ ! -f "Gemfile.lock" ]; then
    echo "📦 Installing Jekyll dependencies..."
    bundle install
else
    echo "📦 Checking dependencies..."
    bundle check || bundle install
fi

echo ""
echo "✨ Starting Jekyll server..."
echo "   Open http://localhost:4000 in your browser"
echo "   Press Ctrl+C to stop the server"
echo ""

# Run Jekyll server with live reload
bundle exec jekyll serve --livereload --host 0.0.0.0
