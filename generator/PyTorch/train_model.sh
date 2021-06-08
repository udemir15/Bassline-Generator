#!/bin/bash


#SBATCH --job-name=NeuralBasslineGenerator
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition mid
#SBATCH --account=users
#SBATCH --qos=users
#SBATCH --time=4:00:00
#SBATCH --output=trainmodel-%J.log
#SBATCH --gres=gpu:tesla_t4:1
##SBATCH --mem-per-cpu=16G
#SBATCH --mem 100000

# Load Anaconda
echo "======================="
echo "Loading Anaconda Module..."
module load anaconda/3.6
module load ffmpeg/4.1.3
module load gcc/9.1.0 
module load cudnn/8.1.1/cuda-11.X
CONDA_ENV=root_env
#conda env create -f environment.yml
source activate $CONDA_ENV
pip install librosa

# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

python train_Seq2SeqGRUWithAttention.py
#python train_Seq2SeqGRUSimple.py