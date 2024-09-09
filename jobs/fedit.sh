#!/bin/bash
#SBATCH --account=project_2009050
#SBATCH --job-name=fedit
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/fedit.out
#SBATCH --error=logs/fedit.err

module --force purge
module load pytorch
source /projappl/project_2009050/torch/bin/activate
cd /projappl/project_2009050/code/mira

export PYTHONPATH=$PYTHONPATH:/projappl/project_2009050/torch/lib/python3.9/site-packages
echo "Current PYTHONPATH: $PYTHONPATH"
python -c "import rouge; print('rouge module is installed and importable')"

srun python main.py --fname ./configs/fedit/configs_instruct.yaml
