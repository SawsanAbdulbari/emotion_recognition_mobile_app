#!/bin/bash
# Script to run a local smoke test with 1 epoch

# Ensure necessary directories exist
mkdir -p logs
mkdir -p configs
mkdir -p outputs/logs
mkdir -p models
mkdir -p outputs/metrics
mkdir -p outputs/checkpoints

# Verify config file exists
if [ ! -f "configs/phase1.yaml" ]; then
    echo "Config file configs/phase1.yaml not found!"
    exit 1
fi

# Models to test locally
MODELS=("mobilenet_v3_small" "mobilevit_xxs")

for MODEL in "${MODELS[@]}"; do
    echo "Running smoke test for $MODEL"
    
    # Create model-specific directories
    mkdir -p outputs/logs/$MODEL
    mkdir -p models/$MODEL
    mkdir -p outputs/metrics/$MODEL
    mkdir -p outputs/checkpoints/$MODEL
    
    # Run with 1 epoch as smoke test
    python train_phase1.py \
        --config configs/phase1.yaml \
        --model $MODEL \
        --pretrained \
        --smoke-test \
        --test
    
    echo "Completed smoke test for $MODEL"
    echo "----------------------------------------"
done

echo "All smoke tests completed!"