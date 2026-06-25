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

BUNDLER_VERSION="${BUNDLER_VERSION:-}"
if [ -z "$BUNDLER_VERSION" ] && [ -f "Gemfile.lock" ]; then
    BUNDLER_VERSION=$(awk '/^BUNDLED WITH$/ { getline; gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print; exit }' Gemfile.lock)
fi
BUNDLER_VERSION="${BUNDLER_VERSION:-2.5.23}"

bundle_cmd() {
    BUNDLER_VERSION="$BUNDLER_VERSION" bundle "$@"
}

# Check if Bundler is installed
if ! command -v bundle &> /dev/null; then
    echo "📦 Installing Bundler $BUNDLER_VERSION..."
    gem install bundler -v "$BUNDLER_VERSION"
elif ! bundle_cmd -v &> /dev/null; then
    echo "📦 Installing Bundler $BUNDLER_VERSION..."
    gem install bundler -v "$BUNDLER_VERSION"
fi

bundle_cmd config set path vendor/bundle

# Install dependencies
if [ ! -f "Gemfile.lock" ]; then
    echo "📦 Installing Jekyll dependencies..."
    bundle_cmd install
else
    echo "📦 Checking dependencies..."
    bundle_cmd check || bundle_cmd install
fi

echo ""
echo "✨ Starting Jekyll server..."
echo "   Open http://localhost:4000 in your browser"
echo "   Press Ctrl+C to stop the server"
echo ""

# Run Jekyll server with live reload
bundle_cmd exec jekyll serve --livereload --host 0.0.0.0
