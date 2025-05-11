#!/bin/bash
#SBATCH --account=project_2014146
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:v100:1,nvme:50
#SBATCH --array=0-2
#SBATCH --output=%A_%a_emotion_phase2.out
#SBATCH --error=%A_%a_emotion_phase2.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sawsan23000@student.hamk.fi

# Define locations
PROJECT_DIR="/projappl/project_2014146/sawsan-abdulbari/projects/emotion_recognition_mobile_app"
DATA_DIR="/scratch/project_2014146/sawsan-abdulbari/data/rafdb_dataset"
LABEL_FILE="/scratch/project_2014146/sawsan-abdulbari/data/rafdb_dataset/processed/labels.csv"
CONFIG_DIR="$PROJECT_DIR/configs"
OUTPUT_DIR="$PROJECT_DIR/models/phase2"
LOG_DIR="$PROJECT_DIR/outputs/logs/phase2"
METRICS_DIR="$PROJECT_DIR/outputs/metrics/phase2"
CHECKPOINTS_DIR="$PROJECT_DIR/outputs/checkpoints/phase2"

# Create necessary directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR
mkdir -p $METRICS_DIR
mkdir -p $CHECKPOINTS_DIR

# Map array task ID to model
MODELS=(emotion_attention_net emotion_attention_net_large emotion_attention_net_resnetls)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
echo "Task $SLURM_ARRAY_TASK_ID → training $MODEL"

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

# Install required packages
pip install --user omegaconf tqdm scikit-learn pandas torchvision matplotlib

# Get current timestamp for output directories
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
echo "Run timestamp: $TIMESTAMP"

# Copy script, config and data to local fast storage
cp $PROJECT_DIR/src/train_phase2.py $LOCAL_SCRATCH/
cp $CONFIG_DIR/phase2.yaml $LOCAL_SCRATCH/
mkdir -p $LOCAL_SCRATCH/raf-db
echo "Copying dataset to local storage..."
rsync -a $DATA_DIR/ $LOCAL_SCRATCH/raf-db/
echo "Dataset copying complete"


# Run the training script with config file and optimizations
srun python train_phase2.py \
    --config phase2.yaml \
    --data_dir $LOCAL_SCRATCH/raf-db \
    --label_file $LABEL_FILE \
    --output_dir $OUTPUT_DIR/$MODEL \
    --model $MODEL \
    --seed 42 \
    --num_workers $SLURM_CPUS_PER_TASK \
    --mixed_precision \
    --pin_memory \
    --checkpoint_freq 5

# Copy all results back to persistent storage
echo "Copying results back to persistent storage..."
mkdir -p $OUTPUT_DIR/$MODEL

# Copy the model directory structure with timestamps
rsync -av $LOCAL_SCRATCH/${MODEL}_* $OUTPUT_DIR/ 2>/dev/null || true

# Also copy individual files in case they weren't saved with timestamp prefixes
rsync -av $LOCAL_SCRATCH/*.pt $OUTPUT_DIR/$MODEL/ 2>/dev/null || true
rsync -av $LOCAL_SCRATCH/*.json $OUTPUT_DIR/$MODEL/ 2>/dev/null || true
rsync -av $LOCAL_SCRATCH/*.csv $OUTPUT_DIR/$MODEL/ 2>/dev/null || true

# Copy logs and checkpoints
rsync -av $LOCAL_SCRATCH/runs/* $LOG_DIR/$MODEL/ 2>/dev/null || true
rsync -av $LOCAL_SCRATCH/checkpoints/* $CHECKPOINTS_DIR/$MODEL/ 2>/dev/null || true

# Copy metrics
rsync -av $LOCAL_SCRATCH/metrics/* $METRICS_DIR/$MODEL/ 2>/dev/null || true

echo "Job completed successfully!" 