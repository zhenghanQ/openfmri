#!/bin/bash

python dicomconvert2.py \
	-d /mindhive/xnat/dicom_storage/voice/%s/*/* \
	-s voice_999b voice_999_20150304 voice_986 voice_899 \
	-f heuristic.py -o /om/project/voice/mri/openfmri
