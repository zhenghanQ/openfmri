#!/bin/bash
mkdir temp
temp_dir=/om/user/zqi/projects/CASL/Analysis/bids/openfmri/convert/temp
for subj in `ls -d /mindhive/xnat/dicom_storage/CASL/CASL*`
do
subj=$(basename $subj)
mkdir ${temp_dir}/${subj}_dicom
cd ${temp_dir}/${subj}_dicom
echo '#!bin/bash' > dicom_run.sh
echo "heudiconv -d /mindhive/xnat/dicom_storage/CASL/%s/dicom/*.dcm -o /om/user/zqi/projects/CASL/Results/Imaging/openfmri -f /om/user/zqi/projects/CASL/Analysis/bids/openfmri/convert/heuristic_CASL_bids.py -c dcm2niix -q om_all_nodes -s ${subj} -b" >> dicom_run.sh
done
for subj in `ls -d /mindhive/xnat/dicom_storage/CASL/CASL*`
do
subj=$(basename $subj)
cd ${temp_dir}/${subj}_dicom
bash dicom_run.sh
done

