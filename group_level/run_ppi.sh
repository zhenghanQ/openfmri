#!/bin/bash
outdir=/om/project/voice/processedData/groupAnalysis/ppi_hc
openfmri_dir=/om/project/voice/processedData/openfmri/
ppidir=/om/project/voice/processedData/l1analysis/ppi/
workdir=/om/scratch/Tue/$USER/ppi_hc

python group_onesample_openfmri.py -m 1 -t 5 \
  -o $outdir \
  -d $openfmri_dir \
  --l1 $ppidir \
  -w $workdir \
  --plugin "SLURM"
