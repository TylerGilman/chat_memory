#!/bin/bash

# Check if NVIDIA Docker runtime is installed
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA drivers not found. Please install NVIDIA drivers first."
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo "Please log out and back in for Docker permissions to take effect."
    exit 1
fi

# Check if nvidia-docker is installed
if ! docker info | grep -i "runtime: nvidia" &> /dev/null; then
    echo "Installing NVIDIA Docker runtime..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update && sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker
fi

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Please set your Hugging Face token:"
    read -s HF_TOKEN
    export HF_TOKEN=$HF_TOKEN
fi

# Build and run the container
docker-compose up --build
