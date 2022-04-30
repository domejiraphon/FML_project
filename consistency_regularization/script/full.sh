#!/bin/bash
#SBATCH --job-name=js_sn
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30GB
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
##SBATCH --mail-type=BEGIN
##SBATCH --mail-user=jy3694@nyu.edu

cat<<EOF
Job starts at: $(date)

EOF

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

singularity exec --nv  --overlay /scratch/jy3694/torchenv.ext3:ro \
    /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
    /bin/bash -c "source /ext3/env.sh; python cr_train.py --model_dir js_sn --use_sn --loss_func JS --gpus 1"