#!/bin/bash

roi=Right-Amygdala

openfmri_dir=/om/project/voice/processedData/openfmri/
out_dir=/om/project/voice/processedData/l1analysis/ppi/$roi
session_id=session001

task=5
for s in ${openfmri_dir}voice97*; do
	sub=${s: -8}
	echo $sub
	SUBJECTS_DIR=/om/project/voice/processedData/fsdata/$sub/
	python fmri_ants_openfmri_sparse_ppi.py \
	-d $openfmri_dir \
	-m 001 \
	-x $sub \
	--sd /om/project/voice/processedData/fsdata/$sub/ \
	-o $out_dir \
	-w /om/scratch/Thu/ksitek/ppi/$roi/task00${task}/$sub \
	-t $task \
	--session_id $session_id \
	--plugin "SLURM"
done
