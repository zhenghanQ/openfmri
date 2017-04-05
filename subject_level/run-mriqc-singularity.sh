#!/bin/bash

# last updated: 3/3/2017

base=/storage/gablab001/banda

# first go to data directory and grab array of all subjects
data=$base/data/
pushd $data
subjs=$(ls sub-* -d -1 | tr '\n' ' ')
popd

# path to image
IMG_PATH='/storage/gablab001/data/singularity-images/poldracklab_mriqc_0.9.1-2017-03-22-660a7e8ea375.img' # untested

# mriqc args
WORKDIR='/mnt/workdir/mathias/qa_workdir'
DATADIR='/mnt/data'
OUTDIR='/mnt/data/derivatives/mriqc'

# run each subject individually

for subj in $subjs
do
# command to run (NOTE: afni path added because currently afni is not loaded properly)
sbatch -t 23:00:00 --mem 10GB singularity exec -B $base:/mnt -c $IMG_PATH bash -c "PATH=/usr/lib/afni/bin/:\$PATH mriqc --participant_label $subj --n_procs 10 --mem_gb 10 -w $WORKDIR $DATADIR $OUTDIR participant"
done
