#!/bin/bash
#SBATCH --account=project_2014146
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --array=0-2
#SBATCH --output=%A_%a_emotion.out
#SBATCH --error=%A_%a_emotion.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sawsan23000@student.hamk.fi

# Define locations
PROJECT_DIR="/projappl/project_2014146/sawsan-abdulbari/projects/emotion_recognition_mobile_app"
DATA_DIR="/scratch/project_2014146/sawsan-abdulbari/data/fer2013_dataset/processed/processed"
LOG_DIR="$PROJECT_DIR/outputs/logs"
mkdir -p $LOG_DIR

# Map array task ID to model
MODELS=(efficientnet_b0 resnet50 mobilenet_v3_small)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
echo "Task $SLURM_ARRAY_TASK_ID → training $MODEL"

# Load CSC modules
module purge
module load pytorch/1.12.1-cuda11.3

# Create model-specific directories
mkdir -p $PROJECT_DIR/outputs/logs/$MODEL \
         $PROJECT_DIR/models/$MODEL \
         $PROJECT_DIR/outputs/metrics/$MODEL \
         $PROJECT_DIR/outputs/checkpoints/$MODEL

# Install required packages if not already installed
pip install --user omegaconf tqdm
# Use the modified script instead of the original one
srun python $SLURM_TMPDIR/train_phase1_fixed.py \
     --config configs/phase1.yaml \
     --model $MODEL \
     --pretrained \
     --data-dir $DATA_DIR \
     --batch-size 32 \
     --log-dir $PROJECT_DIR/outputs/logs/$MODEL \
     --output-dir $PROJECT_DIR/models/$MODEL