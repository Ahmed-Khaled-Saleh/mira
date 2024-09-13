#!/bin/bash
#SBATCH --account=project_2009050
#SBATCH --job-name=puhti_fedit
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=100G
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/puhti_fedit.out
#SBATCH --error=logs/puhti_fedit.err

module --force purge
module load pytorch
source /projappl/project_2009050/python_envs/torch/bin/activate
cd /projappl/project_2009050/code/mira

export PYTHONPATH=$PYTHONPATH:/projappl/project_2009050/python_envs/torch/lib/python3.9/site-packages
echo "Current PYTHONPATH: $PYTHONPATH"
python -c "import rouge; print('rouge module is installed and importable')"

srun python main.py --fname ./configs/fedit/configs_instruct.yaml
