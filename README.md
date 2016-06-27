This repository aggregates Gablab scripts that operate in the OpenfMRI framework.


# group_level

**group_multregress_openfmri.py**

First create a "groups" directory in your openfmri directory. This groups directory needs to contain 3 text files:

(1) **participant_key.txt**
  * A list of the subjects you want to include in your group analysis
  * The first row contains column labels (the first label should be "ids", then the rest should be task numbers)
  * Tab-delimited

Example:
```
ids        task001  task002
SUB_001    1        1
SUB_002    1        1
```

(2) **behav.txt**
  * Contains your regressors of interest, like groups, subject background info, behavioral data 
  * The first row contains column labels (the first label should be "ids", then the rest should be the names of your regressors)
  * Tab-delimited
  * Note: the script will automatically demean your non-group regressors

Example:
```
ids        all     control   asd    gender     wasi_1_vocab_raw
SUB_001    1       1         0      0          3.0
SUB_002    1       0         1      1          -2.3 
```      
   
(3) **contrasts.txt**
  * This file contains 4 columns: task number, contrast description, list of regressors, contrast weights
  * The list of regressors should correspond to the column labels in your behav.txt file
  * Space-delimited

Example:
```
task001 asd_gt_control ['asd','control'] 1 -1
task002 wasi_1_vocab_raw_corr ['all','WASI_1_vocab_raw'] 0 1
```

**Specifying options:**
```
python group_multregress_openfmri.py -m [model #] -t [task #] -o /path/to/output/dir/ -d /path/to/openfmri/dir/ 
-l1 /path/to/l1output/dir/ -w /path/to/working/dir/ -p [plugin] --plugin_args [plugin arguments] [--norev]
```

Or type in `python group_multregress_openfmri.py` at the command line to see usage. Specify the --norev option if reversal of contrasts is already in your contrasts.txt file; otherwise, the script will include outputs for reversal of contrasts by default.

Example:
```
python group_multregress_openfmri.py -m 1 -t 1 -o /om/project/simba/analysis/openfmri/l2output/ 
-d /om/project/simba/analysis/openfmri/ -l1 /om/project/simba/analysis/openfmri/l1output/
-w /om/scratch/Sun/annepark/SIMBA/fmri/ -p 'SLURM' --plugin_args "{'sbatch_args':'-p om_all_nodes'}"
```


# resting_state

**rsfmri_vol_surface_preprocessing_nipy.py**

Resting state preprocessing script, originally from <https://github.com/nipy/nipype/blob/master/examples/rsfmri_vol_surface_preprocessing_nipy.py> (extended here to include TOPUP option)
  * Preprocessing: simultaneous motion and slice-timing correction, regressing out noise (motion parameters, composite norm, outliers, physiological noise through anatomical CompCor), bandpass filtering (.01-.1 Hz), spatial smoothing (6 mm FWHM), normalization to MNI template.
  * Also extracts mean time series from the DKT atlas labels, and maps from volume to FreeSurfer target surfaces (e.g. fsaverage). 
  * Includes option for geometric distortion correction using [FSL's TOPUP tool](http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/TOPUP/).
  * Requires 2mm subcortical atlas and templates available from [mindboggle](http://www.mindboggle.info/data.html), placed in the same directory as the resting state script:
      * Joint Fusion Atlas: OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_MNI152_2mm_v2.nii.gz
      * MNI Template: OASIS-30_Atropos_template_in_MNI152_2mm.nii.gz

**Specifying options:**

```
usage: rsfmri_vol_surface_preprocessing_nipy.py 
    [-h] [-d DICOM_FILE] -f FILES
    [FILES ...] -t TARGET_FILE -s
    SUBJECT_ID --subjects_dir
    FSDIR
    [--target_surfaces TARGET_SURFS [TARGET_SURFS ...], default: 'fsaverage5']
    [--TR TR]
    [--slice_times SLICE_TIMES [SLICE_TIMES ...]]
    [--topup_dicom TOPUP_DICOM]
    [--topup_AP TOPUP_AP]
    [--topup_PA TOPUP_PA]
    [--rest_pe_dir REST_PE_DIR]
    [--vol_fwhm VOL_FWHM, default: 6mm]
    [--surf_fwhm SURF_FWHM, default: 15mm]
    [-l LOWPASS_FREQ, default: 0.1]
    [-u HIGHPASS_FREQ, default: 0.01] -o SINK
    [-w WORK_DIR] [-p PLUGIN, default: 'Linear']
    [--plugin_args PLUGIN_ARGS]
```

Flags in brackets are optional. The dicom file is used to extract information about the resting state time series like TR, slice times, and slice thickness. For non-Siemens dicoms, provide slice times `--slice_times` instead of dicom file `-d`, since the dicom extractor is not guaranteed to work.

Without TOPUP:
```
python rsfmri_vol_surface_preprocessing_nipy.py -d /path/to/example/resting/dicom/ -f /path/to/resting/nifti/ 
-t /path/to/MNI/template [best to use OASIS template] -s [subject ID] --subjects_dir /path/to/fs/dir/ 
--target_surfaces [FreeSurfer target surfaces] -o /path/to/output/dir/ -w /path/to/working/dir/ -p [plugin] 
--plugin_args [plugin arguments]
```

With TOPUP:
```
python rsfmri_vol_surface_preprocessing_nipy.py -d /path/to/example/resting/dicom/ -f /path/to/resting/nifti/ 
-t /path/to/MNI/template [best to use OASIS template] -s [subject ID] --subjects_dir /path/to/fs/dir/ 
--target_surfaces 'fsaverage' -o /path/to/output/dir/ -w /path/to/working/dir/ -p [plugin] --topup_dicom 
/path/to/topup/dicom/ --topup_AP /path/to/AP/topup/nifti/ --topup_PA /path/to/PA/topup/nifti --rest_pe_dir 
[phase-encoding direction of resting time series: AP or PA] --plugin_args [plugin arguments]
```

**TOPUP**
Running the script with TOPUP requires the following inputs: --topup_dicom, --topup_AP, --topup_PA, --rest_pe_dir. Note that you can only run TOPUP if you collected TOPUP images in opposing phase-encoding directions; you do not need to have resting state in both directions.
  * The TOPUP dicom file is used to extract information from the header for calculating the TOPUP images' readout time. 
  * If you collected multiple TOPUP volumes, `--topup_AP` and `--topup_PA` can take merged 4D files, but currently only extracts the first volume from each.
  * `--rest_pe_dir` The phase-encoding direction of your resting state time series: AP or PA. You can check the phase-encoding direction through visual inspection: AP if the image is compressed, PA if the image is stretched, esp. in the frontal and temporal regions. 

Specific example (with TOPUP):
python rsfmri_vol_surface_preprocessing_nipy.py -d /mindhive/xnat/dicom_storage/[project]/[subid]/dicom/*-13-1.dcm 
-f /om/project/[project]/subjects/[subid]/resting/resting_001.nii.gz -t OASIS-30_Atropos_template_in_MNI152_2mm.nii.gz 
-s [subid] --subjects_dir /mindhive/xnat/surfaces/[project]/ -o rest_output -w rest_wd -p 'SLURM' --topup_dicom 
/mindhive/xnat/dicom_storage/[project]/[subid]/dicom/*-22-1.dcm --topup_AP 
/om/project/[project]/subjects/[subid]/topup_rest/topup_rest_001_AP.nii.gz 
--topup_PA /om/project/[project]/subjects/[subid]/topup_rest/topup_rest_001_PA.nii.gz 
--rest_pe_dir AP --plugin_args "{'sbatch_args':'-p om_all_nodes --mem=6GB'}"

**Notes:**
  * If using an SMS sequence, for now you'll need to comment out lines 755-757 (the slice_times, tr, and slice_info inputs of the SpaceTimeRealigner node) - by doing this, only motion correction will be performed, not simultaneous motion and slice-timing correction.
  * TOPUP automatically removes the top slice if your images have an odd number of slices. 
  * For ART, the script uses a composite norm threshold of 1mm for motion, and 3 SD for the intensity Z-threshold. 
