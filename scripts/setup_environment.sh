#!/bin/bash
set -e

echo "Setting up development environment..."

# Remove existing virtual environment if it exists
if [ -d ".trainvenv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf .trainvenv
fi

# Create virtual environment
echo "Creating new virtual environment..."
python3 -m venv .trainvenv

# Make sure the activate script is executable
chmod +x .trainvenv/bin/activate

# Activate virtual environment
source .trainvenv/bin/activate

# Verify activation
if [ "$VIRTUAL_ENV" = "" ]; then
    echo "Error: Virtual environment activation failed"
    exit 1
fi

echo "Virtual environment activated successfully"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Make all scripts executable
echo "Making scripts executable..."
chmod +x scripts/*.sh

echo "Environment setup complete!"
echo "To activate the environment in a new terminal, run: source .trainvenv/bin/activate"
