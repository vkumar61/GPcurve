#!/bin/bash
#SBATCH -t 7-00:00
#SBATCH -o _job0_.out
#SBATCH -e _job0_.err

module purge    
module load anaconda/py3
python main.py 0