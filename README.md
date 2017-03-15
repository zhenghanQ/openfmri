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

# Dicom conversion
## get dicominfo.txt 
Start out running heudiconv without any converter, just passing in dicoms.
'''
heudiconv -d $DICOMPATH/%s/*.dcm -f convertall.py -c none -s $YOUR_SUBJECT
'''
## generate a heuristic file according to the dicominfo.txt
* [example]( https://github.com/nipy/heudiconv/blob/master/heuristics/cmrr_heuristic.py)

## run heudiconv
```
heudiconv -d dicoms_dir -o nifti_dir -f heuristic.py -c dcm2niix -q om_interactive -s $SUBJECT -b
```

## merge longitudinal sessions (ses-pre; ses-post) into one subject folder, change file names to include session info
```
python merge_session.py
```

## create participants.tsv; dataset_description.json; task_bold.json in the same folder of the nifti files

## validate bids data structure
[bids validator](https://github.com/INCF/bids-validator)
__Note: because the validator works either on a local terminal or a local browser, use sshfs to mount the remote data directories to local computer.__
