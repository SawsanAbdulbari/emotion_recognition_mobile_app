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
PROJECT_DIR="/scratch/project_2014146/$USER"
DATA_DIR="/scratch/project_2014146/sawsan-abdulbari/data/fer2013_dataset/processed/processed"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p $LOG_DIR

# Map array task ID to model
MODELS=(efficientnet_b0 mobilenet_v3_small resnet50 densenet121 mobilevit_xxs)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
echo "Task $SLURM_ARRAY_TASK_ID → training $MODEL"

# Load CSC modules
module purge
module load Python/3.8.10
module load pytorch/1.12           # includes torch & torchvision
module load py-matplotlib          # matplotlib bindings
module load py-pandas              # pandas
module load py-scikit-learn        # scikit-learn
 
# ensure --user installs go into ~/.local/
pip install --user omegaconf
# now run
pip install --user timm


# Create model-specific directories
mkdir -p $PROJECT_DIR/outputs/logs/$MODEL \
         $PROJECT_DIR/models/$MODEL \
         $PROJECT_DIR/outputs/metrics/$MODEL \
         $PROJECT_DIR/outputs/checkpoints/$MODEL

# Run the training script
srun python src/train_phase1.py \
     --config configs/phase1.yaml \
     --model $MODEL \
     --pretrained \
     --data-dir $DATA_DIR \
     --log-dir outputs/logs/$MODEL \
     --output-dir models/$MODEL
