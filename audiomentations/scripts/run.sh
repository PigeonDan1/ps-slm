#!/bin/bash
. ./path.sh

nj=4
jobname=audio_aug
image="docker.v2.aispeech.com/hpc/ai_on_device_base-atec:0.0.2"
cpu_cmd="vc submit --image $image --partition pdcpu --mem-per-task 1G --cpu-per-task 1 -pj test --sync"

. utils/parse_options.sh

mkdir -p slurm
param=$@

$cpu_cmd --job $jobname JOB=1:${nj} slurm/${jobname}.JOB.log --cmd "python -u -m audiomentations.launch $param" || exit 1
