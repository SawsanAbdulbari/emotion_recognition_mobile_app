#!/bin/bash
#SBATCH --account=project_ID
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:v100:1,nvme:10
#SBATCH --array=0-4
#SBATCH --output=%A_%a_emotion.out
#SBATCH --error=%A_%a_emotion.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=example@mail.com

# Define locations
PROJECT_DIR="/path/to/project/folder"
DATA_DIR="/path/to/data/folder"
LOG_DIR="$PROJECT_DIR/outputs/logs"
mkdir -p $LOG_DIR

# Map array task ID to model
MODELS=(efficientnet_b0 resnet50 mobilenet_v3_small mobilevit_xxs densenet121)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
echo "Task $SLURM_ARRAY_TASK_ID → training $MODEL"

# Load CSC modules
module purge
module load pytorch/1.12

# Verify python is available
which python
python --version

# Install required packages
pip install --user omegaconf tqdm

# Copy script to local fast temp storage
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
     --output-dir $PROJECT_DIR/models/$MODEL \
     --test

# Copy any remaining results back to persistent storage
cp -r $LOCAL_SCRATCH/* $SLURM_SUBMIT_DIR/ 2>/dev/null || true
