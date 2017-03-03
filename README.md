#BIDS analysis scripts

##To use the old openfmri format scripts, go to the [openfmri branch](https://github.com/gablab/openfmri/tree/openfmri) of this repo

#set up a python environment dedicated for the project
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

##install python packages
* heudiconv: https://github.com/nipy/heudiconv
* dcmstack: https://github.com/moloney/dcmstack
* pybids: https://github.com/INCF/pybids

#Dicom conversion
##get dicominfo.txt 
Start out running heudiconv without any converter, just passing in dicoms.
'''
heudiconv -d $DICOMPATH/%s/*.dcm -f convertall.py -c none -s $YOUR_SUBJECT
'''
##generate a heuristic file according to the dicominfo.txt
*[example]( https://github.com/nipy/heudiconv/blob/master/heuristics/cmrr_heuristic.py)

##run heudiconv
```
heudiconv -d dicoms_dir -o nifti_dir -f heuristic.py -c dcm2niix -q om_interactive -s $SUBJECT -b
```

##
##validate bids data structure
[bids validator](https://github.com/INCF/bids-validator)
