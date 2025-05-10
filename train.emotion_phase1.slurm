#!/bin/bash
#SBATCH --account=project_2014146
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:v100:1,nvme:10G
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

# Load CSC module
module purge
module load pytorch/1.12.1

# Copy script to local fast storage
cp $PROJECT_DIR/src/train_phase1.py $LOCAL_SCRATCH/
cd $LOCAL_SCRATCH

# Create model-specific directories
mkdir -p $PROJECT_DIR/outputs/logs/$MODEL \
         $PROJECT_DIR/models/$MODEL \
         $PROJECT_DIR/outputs/metrics/$MODEL \
         $PROJECT_DIR/outputs/checkpoints/$MODEL

# Run the training script
srun python $LOCAL_SCRATCH/train_phase1.py \
     --config $PROJECT_DIR/configs/phase1.yaml \
     --model $MODEL \
     --pretrained \
     --data-dir $DATA_DIR \
     --batch-size 32 \
     --log-dir $PROJECT_DIR/outputs/logs/$MODEL \
     --output-dir $PROJECT_DIR/models/$MODEL

# Copy any remaining results back to persistent storage
# (In case the script writes any extra files to current directory)
cp -r $LOCAL_SCRATCH/* $SLURM_SUBMIT_DIR/ 2>/dev/null || true