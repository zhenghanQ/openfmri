#!/bin/bash

for sub in voice_984 voice_983 voice_982 voice_981 voice_980 voice_979 voice_979b; do
#for subdir in /mindhive/xnat/dicom_storage/voice/voice_*; do
	#sub=$(basename $subdir)
	rm dcmconv_${sub}.sh
	echo "#!/bin/bash
	python dicomconvert2.py \
		-d /mindhive/xnat/dicom_storage/voice/%s/*/* \
		-s $sub \
		-f heuristic_example.py \
		-o /om/project/voice/RawData/voice/" >> dcmconv_${sub}.sh

	sbatch --mem=20G dcmconv_${sub}.sh
done
