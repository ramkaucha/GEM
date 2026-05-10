#!/bin/bash

#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=5:00:00
#PBS -N ecg_img_gen

cd ~/ra/GEM/gem_generation/ecg-image-generator

source /srv/scratch/z5367751/miniconda3/bin/activate gem

python gen_ecg_images_from_data_batch.py -i /srv/scratch/z5367751/records/ -o /srv/scratch/z5367751/records_img
