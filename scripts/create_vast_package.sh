#!/bin/bash
set -e

echo "Creating vast.ai package..."

# Create temporary directory
TEMP_DIR="vast_package"
rm -rf $TEMP_DIR
mkdir -p $TEMP_DIR

# Copy necessary files
cp Dockerfile $TEMP_DIR/
cp requirements.txt $TEMP_DIR/
cp scripts/setup_vast.sh $TEMP_DIR/
cp -r training $TEMP_DIR/

# Copy data
mkdir -p $TEMP_DIR/data
cp data/train/train.jsonl $TEMP_DIR/data/
cp data/test/test.jsonl $TEMP_DIR/data/

# Create tar file
cd $TEMP_DIR
tar -czf ../chat_memory.tar.gz *
cd ..

# Clean up
rm -rf $TEMP_DIR

echo "Package chat_memory.tar.gz created successfully"
