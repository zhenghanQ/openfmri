#!/bin/bash

for roi_path in /om/project/voice/processedData/l1analysis/ppi/*; do
  roi=${roi_path##*/}

  outdir=/om/project/voice/processedData/groupAnalysis/ppi_hc/$roi
  openfmri_dir=/om/project/voice/processedData/openfmri/
  ppidir=$roi_path
  workdir=/om/scratch/Fri/$USER/ppi_hc/$roi

  python group_onesample_openfmri.py -m 1 -t 5 \
    -o $outdir \
    -d $openfmri_dir \
    --l1 $ppidir \
    -w $workdir \
    --plugin "SLURM"
done
