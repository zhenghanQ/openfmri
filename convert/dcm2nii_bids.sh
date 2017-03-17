#!/bin/bash
mkdir temp
temp_dir=/om/user/zqi/projects/CASL/Analysis/bids/openfmri/convert/temp
for subj in `ls -d /mindhive/xnat/dicom_storage/CASL/CASL*`
do
subj=$(basename $subj)
mkdir ${temp_dir}/${subj}_dicom
cd ${temp_dir}/${subj}_dicom
echo '#!/bin/bash' > dicom_run.sh
echo "singularity exec -B /mindhive/xnat/dicom_storage/CASL/:/dicomdir -B /om/user/zqi/projects/CASL/Results/Imaging/openfmri/:/output -B /om/user/zqi/projects/CASL/Analysis/bids/openfmri/convert/:/mnt -c /storage/gablab001/data/singularity-images/heudiconv/nipy/heudiconv heudiconv -d /dicomdir/%s/dicom/*.dcm -c dcm2niix -o /output -f /mnt/heuristic_CASL_bids.py -s ${subj} -b" >> dicom_run.sh
done
for subj in `ls -d /mindhive/xnat/dicom_storage/CASL/CASL*`
do
subj=$(basename $subj)
cd ${temp_dir}/${subj}_dicom
sbatch -t 2:00:00 --mem=20GB -c2 dicom_run.sh
done

