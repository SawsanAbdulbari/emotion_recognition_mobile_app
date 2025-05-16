#!/bin/bash
#SBATCH --account=project_ID
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:v100:1,nvme:50
#SBATCH --array=0-2
#SBATCH --output=%A_%a_emotion_phase2_test.out
#SBATCH --error=%A_%a_emotion_phase2_test.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=exampl@mail.com

# Set environment variables to disable hash checking for model weights
export TORCH_HOME=$LOCAL_SCRATCH/.torch
export TORCH_HUB_DISABLE_HASHCHECK=1
export PYTHONWARNINGS="ignore::UserWarning"

# Define locations
PROJECT_DIR="/path/to/project/folder"
DATA_DIR="/path/to/data/folder"
LABEL_FILE="/path/to/data/folder/labels.csv"    
CONFIG_DIR="$PROJECT_DIR/configs"
OUTPUT_DIR="$PROJECT_DIR/models/phase2_test"
LOG_DIR="$PROJECT_DIR/outputs/logs/phase2_test"
METRICS_DIR="$PROJECT_DIR/outputs/metrics/phase2_test"
CHECKPOINTS_DIR="$PROJECT_DIR/outputs/checkpoints/phase2_test"

# Create necessary directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR
mkdir -p $METRICS_DIR
mkdir -p $CHECKPOINTS_DIR

# Map array task ID to model, model names should match src/train_phase2.py
MODELS=(efficientnet_b0 emotion_attention_net emotion_attention_net_resnet)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
echo "TESTING - Task $SLURM_ARRAY_TASK_ID → training $MODEL for 1 epoch only"

# Create model-specific directories
mkdir -p $OUTPUT_DIR/$MODEL
mkdir -p $LOG_DIR/$MODEL
mkdir -p $METRICS_DIR/$MODEL
mkdir -p $CHECKPOINTS_DIR/$MODEL

# Load CSC module
module purge
module load pytorch/2.1

# Verify python is available
which python
python --version

# Install required packages with version constraints to ensure compatibility
pip install --user omegaconf tqdm scikit-learn pandas matplotlib

# Get current timestamp for output directories
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
echo "TEST RUN timestamp: $TIMESTAMP"

# Set up local directories
LOCAL_DATA_DIR="$LOCAL_SCRATCH/raf-db"
LOCAL_CONFIG="$LOCAL_SCRATCH/phase2.yaml"
LOCAL_SCRIPT="$LOCAL_SCRATCH/train_phase2.py"

# Copy script, config and data to local fast storage
echo "Copying script and config files..."
cp $PROJECT_DIR/src/train_phase2.py $LOCAL_SCRIPT
cp $CONFIG_DIR/phase2.yaml $LOCAL_CONFIG

# Check if files were copied successfully
if [ ! -f "$LOCAL_SCRIPT" ]; then
    echo "ERROR: Failed to copy train_phase2.py to $LOCAL_SCRATCH"
    exit 1
fi

if [ ! -f "$LOCAL_CONFIG" ]; then
    echo "ERROR: Failed to copy phase2.yaml to $LOCAL_SCRATCH"
    exit 1
fi

# Create data directory and copy dataset
mkdir -p $LOCAL_DATA_DIR
echo "Copying dataset to local storage..."
rsync -av $DATA_DIR/ $LOCAL_DATA_DIR/
echo "Dataset copying complete"

# Check if the dataset was copied
echo "Checking local data directory contents:"
ls -la $LOCAL_DATA_DIR/
echo "Checking label file existence:"
ls -la $LABEL_FILE || echo "WARNING: Label file not found at $LABEL_FILE"

# Change to local scratch directory
cd $LOCAL_SCRATCH
echo "Working directory: $(pwd)"

# Run the training script with config file and optimizations - ONLY 1 EPOCH
echo "Starting TEST TRAINING for model: $MODEL - 1 EPOCH ONLY"
srun python $LOCAL_SCRIPT \
    --config $LOCAL_CONFIG \
    --data_dir $LOCAL_DATA_DIR \
    --label_file $LABEL_FILE \
    --output_dir $LOCAL_SCRATCH/$MODEL-test-$TIMESTAMP \
    --model $MODEL \
    --epochs 1 \
    --seed 42 \
    --num_workers $SLURM_CPUS_PER_TASK \
    --mixed_precision \
    --pin_memory

# Check if training was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Test training failed for model $MODEL"
    exit 1
fi

# Show what files were generated
echo "Files generated in LOCAL_SCRATCH:"
find $LOCAL_SCRATCH -type f -name "*.pt" -o -name "*.json" -o -name "*.csv" | sort

# Copy all results back to persistent storage
echo "Copying test results back to persistent storage..."
mkdir -p $OUTPUT_DIR/$MODEL/test-$TIMESTAMP

# Copy the model directory with timestamps
if [ -d "$LOCAL_SCRATCH/$MODEL-test-$TIMESTAMP" ]; then
    echo "Copying from $LOCAL_SCRATCH/$MODEL-test-$TIMESTAMP to $OUTPUT_DIR/$MODEL/test-$TIMESTAMP"
    rsync -av $LOCAL_SCRATCH/$MODEL-test-$TIMESTAMP/ $OUTPUT_DIR/$MODEL/test-$TIMESTAMP/
else
    echo "WARNING: Expected output directory $LOCAL_SCRATCH/$MODEL-test-$TIMESTAMP not found"
    
    # Look for model output directories with any timestamp
    MODEL_DIRS=$(find $LOCAL_SCRATCH -maxdepth 1 -type d -name "${MODEL}*" 2>/dev/null)
    if [ ! -z "$MODEL_DIRS" ]; then
        echo "Found alternative model directories, copying those instead:"
        echo "$MODEL_DIRS"
        for dir in $MODEL_DIRS; do
            base_dir=$(basename "$dir")
            rsync -av $dir/ $OUTPUT_DIR/$MODEL/$base_dir/
        done
    fi
fi

echo "Test job completed successfully for model: $MODEL (Task $SLURM_ARRAY_TASK_ID)" 
