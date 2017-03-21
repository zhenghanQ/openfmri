# BIDS analysis scripts

## To use the old openfmri format scripts, go to the [openfmri branch](https://github.com/gablab/openfmri/tree/openfmri) of this repo

# set up a python environment dedicated for the project
* conda clone from an existing python env and rename the env
* add a new function in ~/.projects
```
    function CASL_env()
    {
    source ~/.env/.CASL_environment
    }
```
* add a new .CASL_environment text file in ~/.env/ (backed up in git)
* to update nipype package: 
```
conda config --add channels conda-forge
conda install nipype
```

## install python packages
* heudiconv: https://github.com/nipy/heudiconv
* dcmstack: https://github.com/moloney/dcmstack
* pybids: https://github.com/INCF/pybids
* singularity container is located at: /storage/gablab001/data/singularity-images/heudiconv/nipy/heudiconv

# Dicom conversion
## get dicominfo.txt 
Start out running heudiconv without any converter, just passing in dicoms.
'''
heudiconv -d $DICOMPATH/%s/*.dcm -f convertall.py -c none -s $YOUR_SUBJECT
'''
## generate a heuristic file according to the dicominfo.txt
* [example]( https://github.com/nipy/heudiconv/blob/master/heuristics/cmrr_heuristic.py)

## run heudiconv
* ```heudiconv -d dicoms_dir -o nifti_dir -f heuristic.py -c dcm2niix -q om_interactive -s $SUBJECT -b```
* if using singularity:```singularity exec -B /mindhive/xnat/dicom_storage/CASL/:/dicomdir -B /om/user/zqi/projects/CASL/Results/Imaging/openfmri/:/output -B /om/user/zqi/projects/CASL/Analysis/bids/openfmri/convert/:/mnt -c /storage/gablab001/data/singularity-images/heudiconv/nipy/heudiconv heudiconv -d /dicomdir/%s/dicom/*.dcm -c dcm2niix -s CASL13100 -o /output/testagain -f /mnt/heuristic_CASL_bids.py -b```
* run through all participants: ```bash dcm2nii_bids.sh```

## merge longitudinal sessions (ses-pre; ses-post) into one subject folder, change file names to include session info
```
python merge_session.py
```

## create participants.tsv; dataset_description.json; task_bold.json in the same folder of the nifti files

## validate bids data structure
[bids validator](https://github.com/INCF/bids-validator)
* use the docker image: bids/base_validator
* on openmind: combine singularity with docker
..* create a directory to copy a docker image onto sigularity
```
$ export SINGULARITY_CACHEDIR=$PWD
$ singularity -c shell docker://bids/base_validator
Singularity.base_validator> exit
```
..* rename the singularity file to something meaningful
..* mounting directory to singularity container and run bids validator
```
$ singularity shell -B /om/user/zqi/projects/CASL/Results/Imaging/openfmri/:/mnt -c bids-validator/bids/base_validator/
Singularity.base_validator>/usr/bin/bids-validator /mnt —verbose
```
## [mriqc](http://mriqc.readthedocs.io/en/latest/)
* Either use docker on local computer or signularity on HPC
* on the server, cd into the docker image directory
* the current docker-to-sigularity method does not load afni path correctly
```
singularity exec -B /om:/mnt -c poldracklab_mriqc.img bash -c "PATH=/usr/lib/afni/bin/:\$PATH mriqc --participant_label sub-CASL13100 -w /mnt/scratch/Mon/zqi /mnt/user/zqi/projects/CASL/Results/Imaging/openfmri/ /mnt/user/zqi/projects/CASL/Results/Imaging/qc participant"
```

## [fmriprep](http://fmriprep.readthedocs.io/en/stable/installation.html)
* copy singularity image to HPC
* on the server, cd into the docker image directory
```
singularity exec -B /om:/mnt -c poldracklab_fmriprep_latest-2017-01-13-98bd99012ac2.img fmriprep --participant_label sub-CASL13100 -w /mnt/scratch/Mon/zqi /mnt/user/zqi/projects/CASL/Results/Imaging/openfmri/ -s ses-pre --task-id sent /mnt/user/zqi/projects/CASL/Results/Imaging/fmriprep_out participant
```

