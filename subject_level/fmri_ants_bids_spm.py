#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
=============================================
fMRI: OpenfMRI.org data, FSL, ANTS, c3daffine
=============================================

A growing number of datasets are available on `OpenfMRI <http://openfmri.org>`_.
This script demonstrates how to use nipype to analyze a data set::

    python fmri_ants_openfmri.py --datasetdir ds107
"""

from nipype import config
config.enable_provenance()

from nipype.external import six

from glob import glob
import os

use_spm_smooth = True
use_spm_model = True

import nipype.interfaces.spm as spm
spm.SPMCommand.set_mlab_paths(paths='/cm/shared/openmind/spm/spm12/spm12_r6225/')

import nipype.pipeline.engine as pe
import nipype.algorithms.modelgen as model
import nipype.algorithms.rapidart as ra
import nipype.interfaces.fsl as fsl
import nipype.interfaces.ants as ants
from nipype.algorithms.misc import TSNR
from nipype.interfaces.c3 import C3dAffineTool
import nipype.interfaces.io as nio
import nipype.interfaces.utility as niu
from nipype.workflows.fmri.fsl import (create_featreg_preproc,
                                       create_modelfit_workflow,
                                       create_fixed_effects_flow)

from nipype import LooseVersion
from nipype import Workflow, Node, MapNode
from nipype.interfaces import (fsl, Function, ants, freesurfer)

from nipype.interfaces.utility import Rename, Merge, IdentityInterface
from nipype.utils.filemanip import filename_to_list
from nipype.interfaces.io import DataSink, FreeSurferSource
import nipype.interfaces.freesurfer as fs

version = 0
if fsl.Info.version() and \
    LooseVersion(fsl.Info.version()) > LooseVersion('5.0.6'):
    version = 507

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

imports = ['import os',
           'import nibabel as nb',
           'import numpy as np',
           'import scipy as sp',
           'from nipype.utils.filemanip import filename_to_list, list_to_filename, split_filename',
           'from scipy.special import legendre'
           ]

def median(in_files):
    """Computes an average of the median of each realigned timeseries

    Parameters
    ----------

    in_files: one or more realigned Nifti 4D time series

    Returns
    -------

    out_file: a 3D Nifti file
    """
    average = None
    for idx, filename in enumerate(filename_to_list(in_files)):
        img = nb.load(filename)
        data = np.median(img.get_data(), axis=3)
        if average is None:
            average = data
        else:
            average = average + data
    median_img = nb.Nifti1Image(average/float(idx + 1),
                                img.get_affine(), img.get_header())
    filename = os.path.join(os.getcwd(), 'median.nii.gz')
    median_img.to_filename(filename)
    return filename


def create_reg_workflow(name='registration'):
    """Create a FEAT preprocessing workflow together with freesurfer

    Parameters
    ----------
        name : name of workflow (default: 'registration')

    Inputs:

        inputspec.source_files : files (filename or list of filenames to register)
        inputspec.mean_image : reference image to use
        inputspec.anatomical_image : anatomical image to coregister to
        inputspec.target_image : registration target

    Outputs:

        outputspec.func2anat_transform : FLIRT transform
        outputspec.anat2target_transform : FLIRT+FNIRT transform
        outputspec.transformed_files : transformed files in target space
        outputspec.transformed_mean : mean image in target space

    Example
    -------
    """

    register = pe.Workflow(name=name)

    inputnode = pe.Node(interface=niu.IdentityInterface(fields=['source_files',
                                                                 'mean_image',
                                                                 'anatomical_image',
                                                                 'target_image',
                                                                 'target_image_brain',
                                                                 'config_file']),
                        name='inputspec')
    outputnode = pe.Node(interface=niu.IdentityInterface(fields=['func2anat_transform',
                                                                 'anat2target_transform',
                                                                 'transformed_files',
                                                                 'transformed_mean',
                                                                 'anat2target',
                                                                 'mean2anat_mask'
                                                                 ]),
                         name='outputspec')

    """
    Estimate the tissue classes from the anatomical image. But use spm's segment
    as FSL appears to be breaking.
    """

    stripper = pe.Node(fsl.BET(), name='stripper')
    register.connect(inputnode, 'anatomical_image', stripper, 'in_file')
    fast = pe.Node(fsl.FAST(), name='fast')
    register.connect(stripper, 'out_file', fast, 'in_files')

    """
    Binarize the segmentation
    """

    binarize = pe.Node(fsl.ImageMaths(op_string='-nan -thr 0.5 -bin'),
                       name='binarize')
    pickindex = lambda x, i: x[i]
    register.connect(fast, ('partial_volume_files', pickindex, 2),
                     binarize, 'in_file')

    """
    Calculate rigid transform from mean image to anatomical image
    """

    mean2anat = pe.Node(fsl.FLIRT(), name='mean2anat')
    mean2anat.inputs.dof = 6
    register.connect(inputnode, 'mean_image', mean2anat, 'in_file')
    register.connect(stripper, 'out_file', mean2anat, 'reference')

    """
    Now use bbr cost function to improve the transform
    """

    mean2anatbbr = pe.Node(fsl.FLIRT(), name='mean2anatbbr')
    mean2anatbbr.inputs.dof = 6
    mean2anatbbr.inputs.cost = 'bbr'
    mean2anatbbr.inputs.schedule = os.path.join(os.getenv('FSLDIR'),
                                                'etc/flirtsch/bbr.sch')
    register.connect(inputnode, 'mean_image', mean2anatbbr, 'in_file')
    register.connect(binarize, 'out_file', mean2anatbbr, 'wm_seg')
    register.connect(inputnode, 'anatomical_image', mean2anatbbr, 'reference')
    register.connect(mean2anat, 'out_matrix_file',
                     mean2anatbbr, 'in_matrix_file')

    """
    Create a mask of the median image coregistered to the anatomical image
    """
    
    mean2anat_mask = Node(fsl.BET(mask=True), name='mean2anat_mask')
    register.connect(mean2anatbbr, 'out_file', mean2anat_mask, 'in_file')

    """
    Convert the BBRegister transformation to ANTS ITK format
    """

    convert2itk = pe.Node(C3dAffineTool(),
                          name='convert2itk')
    convert2itk.inputs.fsl2ras = True
    convert2itk.inputs.itk_transform = True
    register.connect(mean2anatbbr, 'out_matrix_file', convert2itk, 'transform_file')
    register.connect(inputnode, 'mean_image',convert2itk, 'source_file')
    register.connect(stripper, 'out_file', convert2itk, 'reference_file')

    """
    Compute registration between the subject's structural and MNI template
    This is currently set to perform a very quick registration. However, the
    registration can be made significantly more accurate for cortical
    structures by increasing the number of iterations
    All parameters are set using the example from:
    #https://github.com/stnava/ANTs/blob/master/Scripts/newAntsExample.sh
    """

    reg = pe.Node(ants.Registration(), name='antsRegister')
    reg.inputs.output_transform_prefix = "output_"
    reg.inputs.transforms = ['Rigid', 'Affine', 'SyN']
    reg.inputs.transform_parameters = [(0.1,), (0.1,), (0.2, 3.0, 0.0)]
    reg.inputs.number_of_iterations = [[10000, 11110, 11110]] * 2 + [[100, 30, 20]]
    reg.inputs.dimension = 3
    reg.inputs.write_composite_transform = True
    reg.inputs.collapse_output_transforms = True
    reg.inputs.initial_moving_transform_com = True
    reg.inputs.metric = ['Mattes'] * 2 + [['Mattes', 'CC']]
    reg.inputs.metric_weight = [1] * 2 + [[0.5, 0.5]]
    reg.inputs.radius_or_number_of_bins = [32] * 2 + [[32, 4]]
    reg.inputs.sampling_strategy = ['Regular'] * 2 + [[None, None]]
    reg.inputs.sampling_percentage = [0.3] * 2 + [[None, None]]
    reg.inputs.convergence_threshold = [1.e-8] * 2 + [-0.01]
    reg.inputs.convergence_window_size = [20] * 2 + [5]
    reg.inputs.smoothing_sigmas = [[4, 2, 1]] * 2 + [[1, 0.5, 0]]
    reg.inputs.sigma_units = ['vox'] * 3
    reg.inputs.shrink_factors = [[3, 2, 1]]*2 + [[4, 2, 1]]
    reg.inputs.use_estimate_learning_rate_once = [True] * 3
    reg.inputs.use_histogram_matching = [False] * 2 + [True]
    reg.inputs.winsorize_lower_quantile = 0.005
    reg.inputs.winsorize_upper_quantile = 0.995
    reg.inputs.float = True
    reg.inputs.output_warped_image = 'output_warped_image.nii.gz'
    reg.inputs.num_threads = 4
    reg.plugin_args = {'qsub_args': '-pe orte 4',
                       'sbatch_args': '--mem=6G -c 4'}
    register.connect(stripper, 'out_file', reg, 'moving_image')
    register.connect(inputnode,'target_image_brain', reg,'fixed_image')


    """
    Concatenate the affine and ants transforms into a list
    """

    pickfirst = lambda x: x[0]

    merge = pe.Node(niu.Merge(2), iterfield=['in2'], name='mergexfm')
    register.connect(convert2itk, 'itk_transform', merge, 'in2')
    register.connect(reg, 'composite_transform', merge, 'in1')


    """
    Transform the mean image. First to anatomical and then to target
    """

    warpmean = pe.Node(ants.ApplyTransforms(),
                       name='warpmean')
    warpmean.inputs.input_image_type = 0
    warpmean.inputs.interpolation = 'Linear'
    warpmean.inputs.invert_transform_flags = [False, False]
    warpmean.inputs.terminal_output = 'file'

    register.connect(inputnode,'target_image_brain', warpmean,'reference_image')
    register.connect(inputnode, 'mean_image', warpmean, 'input_image')
    register.connect(merge, 'out', warpmean, 'transforms')

    """
    Transform the remaining images. First to anatomical and then to target
    """

    warpall = pe.MapNode(ants.ApplyTransforms(),
                         iterfield=['input_image'],
                         name='warpall')
    warpall.inputs.input_image_type = 0
    warpall.inputs.interpolation = 'Linear'
    warpall.inputs.invert_transform_flags = [False, False]
    warpall.inputs.terminal_output = 'file'

    register.connect(inputnode,'target_image_brain',warpall,'reference_image')
    register.connect(inputnode,'source_files', warpall, 'input_image')
    register.connect(merge, 'out', warpall, 'transforms')


    """
    Assign all the output files
    """

    register.connect(reg, 'warped_image', outputnode, 'anat2target')
    register.connect(warpmean, 'output_image', outputnode, 'transformed_mean')
    register.connect(warpall, 'output_image', outputnode, 'transformed_files')
    register.connect(mean2anatbbr, 'out_matrix_file',
                     outputnode, 'func2anat_transform')
    register.connect(mean2anat_mask, 'mask_file',
                     outputnode, 'mean2anat_mask')
    register.connect(reg, 'composite_transform',
                     outputnode, 'anat2target_transform')

    return register

def get_aparc_aseg(files):
    """Return the aparc+aseg.mgz file"""
    for name in files:
        if 'aparc+aseg.mgz' in name:
            return name
    raise ValueError('aparc+aseg.mgz not found')

def create_fs_reg_workflow(name='registration'):
    """Create a FEAT preprocessing workflow together with freesurfer

    Parameters
    ----------

    ::

        name : name of workflow (default: 'registration')

    Inputs::

        inputspec.mean_image : reference image to use
        inputspec.target_image : registration target

    Outputs::

        outputspec.func2anat_transform : FLIRT transform
        outputspec.anat2target_transform : FLIRT+FNIRT transform
        outputspec.transformed_files : transformed files in target space
        outputspec.transformed_mean : mean image in target space

    Example
    -------

    """

    register = Workflow(name=name)

    inputnode = Node(interface=IdentityInterface(fields=['mean_image',
                                                         'subject_id',
                                                         'subjects_dir',
                                                         'target_image']),
                     name='inputspec')

    outputnode = Node(interface=IdentityInterface(fields=['func2anat_transform',
                                                          'out_reg_file',
                                                          'anat2target_transform',
                                                          'transforms',
                                                          'transformed_mean',
                                                          'min_cost_file',
                                                          'anat2target',
                                                          'aparc',
                                                          'mean2anat_mask',
                                                          'mask_file'
                                                          ]),
                      name='outputspec')

    # Get the subject's freesurfer source directory
    fssource = Node(FreeSurferSource(),
                    name='fssource')
    fssource.run_without_submitting = True
    register.connect(inputnode, 'subject_id', fssource, 'subject_id')
    register.connect(inputnode, 'subjects_dir', fssource, 'subjects_dir')

    convert = Node(freesurfer.MRIConvert(out_type='nii'),
                   name="convert")
    register.connect(fssource, 'T1', convert, 'in_file')

    # Coregister the median to the surface
    bbregister = Node(freesurfer.BBRegister(registered_file=True),
                    name='bbregister')
    bbregister.inputs.init = 'fsl'
    bbregister.inputs.contrast_type = 't2'
    bbregister.inputs.out_fsl_file = True
    bbregister.inputs.epi_mask = True
    register.connect(inputnode, 'subject_id', bbregister, 'subject_id')
    register.connect(inputnode, 'mean_image', bbregister, 'source_file')
    register.connect(inputnode, 'subjects_dir', bbregister, 'subjects_dir')

    # Create a mask of the median coregistered to the anatomical image
    mean2anat_mask = Node(fsl.BET(mask=True), name='mean2anat_mask')
    register.connect(bbregister, 'registered_file', mean2anat_mask, 'in_file')

    """
    use aparc+aseg's brain mask
    """

    binarize = Node(fs.Binarize(min=0.5, out_type="nii.gz", dilate=1), name="binarize_aparc")
    register.connect(fssource, ("aparc_aseg", get_aparc_aseg), binarize, "in_file")

    stripper = Node(fsl.ApplyMask(), name ='stripper')
    register.connect(binarize, "binary_file", stripper, "mask_file")
    register.connect(convert, 'out_file', stripper, 'in_file')

    """
    Apply inverse transform to aparc file
    """
    aparcxfm = Node(freesurfer.ApplyVolTransform(inverse=True,
                                                 interp='nearest'),
                    name='aparc_inverse_transform')
    register.connect(inputnode, 'subjects_dir', aparcxfm, 'subjects_dir')
    register.connect(bbregister, 'out_reg_file', aparcxfm, 'reg_file')
    register.connect(fssource, ('aparc_aseg', get_aparc_aseg),
                     aparcxfm, 'target_file')
    register.connect(inputnode, 'mean_image', aparcxfm, 'source_file')

    """
    Apply inverse transform to mask file
    """
    maskxfm = Node(freesurfer.ApplyVolTransform(inverse=True,
                                                transformed_file='mask.nii',
                                                interp='nearest'),
                    name='aparc_mask_inverse_transform')
    register.connect(inputnode, 'subjects_dir', maskxfm, 'subjects_dir')
    register.connect(bbregister, 'out_reg_file', maskxfm, 'reg_file')
    register.connect(binarize, 'binary_file', maskxfm, 'target_file')
    register.connect(inputnode, 'mean_image', maskxfm, 'source_file')

    """
    Convert the BBRegister transformation to ANTS ITK format
    """

    convert2itk = Node(C3dAffineTool(), name='convert2itk')
    convert2itk.inputs.fsl2ras = True
    convert2itk.inputs.itk_transform = True
    register.connect(bbregister, 'out_fsl_file', convert2itk, 'transform_file')
    register.connect(inputnode, 'mean_image',convert2itk, 'source_file')
    register.connect(stripper, 'out_file', convert2itk, 'reference_file')

    """
    Compute registration between the subject's structural and MNI template
    This is currently set to perform a very quick registration. However, the
    registration can be made significantly more accurate for cortical
    structures by increasing the number of iterations
    All parameters are set using the example from:
    #https://github.com/stnava/ANTs/blob/master/Scripts/newAntsExample.sh
    """

    reg = Node(ants.Registration(), name='antsRegister')
    reg.inputs.output_transform_prefix = "output_"
    reg.inputs.transforms = ['Rigid', 'Affine', 'SyN']
    reg.inputs.transform_parameters = [(0.1,), (0.1,), (0.2, 3.0, 0.0)]
    reg.inputs.number_of_iterations = [[10000, 11110, 11110]] * 2 + [[100, 30, 20]]
    reg.inputs.dimension = 3
    reg.inputs.write_composite_transform = True
    reg.inputs.collapse_output_transforms = True
    reg.inputs.initial_moving_transform_com = True
    reg.inputs.metric = ['Mattes'] * 2 + [['Mattes', 'CC']]
    reg.inputs.metric_weight = [1] * 2 + [[0.5, 0.5]]
    reg.inputs.radius_or_number_of_bins = [32] * 2 + [[32, 4]]
    reg.inputs.sampling_strategy = ['Regular'] * 2 + [[None, None]]
    reg.inputs.sampling_percentage = [0.3] * 2 + [[None, None]]
    reg.inputs.convergence_threshold = [1.e-8] * 2 + [-0.01]
    reg.inputs.convergence_window_size = [20] * 2 + [5]
    reg.inputs.smoothing_sigmas = [[4, 2, 1]] * 2 + [[1, 0.5, 0]]
    reg.inputs.sigma_units = ['vox'] * 3
    reg.inputs.shrink_factors = [[3, 2, 1]]*2 + [[4, 2, 1]]
    reg.inputs.use_estimate_learning_rate_once = [True] * 3
    reg.inputs.use_histogram_matching = [False] * 2 + [True]
    reg.inputs.winsorize_lower_quantile = 0.005
    reg.inputs.winsorize_upper_quantile = 0.995
    reg.inputs.float = True
    reg.inputs.output_warped_image = 'output_warped_image.nii.gz'
    reg.inputs.num_threads = 4
    reg.plugin_args = {'qsub_args': '-pe orte 4',
                       'sbatch_args': '--mem=6G -c 4'}
    register.connect(stripper, 'out_file', reg, 'moving_image')
    register.connect(inputnode,'target_image', reg,'fixed_image')


    """
    Concatenate the affine and ants transforms into a list
    """

    pickfirst = lambda x: x[0]

    merge = Node(Merge(2), iterfield=['in2'], name='mergexfm')
    register.connect(convert2itk, 'itk_transform', merge, 'in2')
    register.connect(reg, 'composite_transform', merge, 'in1')


    """
    Transform the mean image. First to anatomical and then to target
    """
    warpmean = Node(ants.ApplyTransforms(), name='warpmean')
    warpmean.inputs.input_image_type = 0
    warpmean.inputs.interpolation = 'Linear'
    warpmean.inputs.invert_transform_flags = [False, False]
    warpmean.inputs.terminal_output = 'file'
    #warpmean.inputs.num_threads = 4
    #warpmean.plugin_args = {'sbatch_args': '--mem=4G -c 4'}


    """
    Assign all the output files
    """

    register.connect(warpmean, 'output_image', outputnode, 'transformed_mean')

    register.connect(inputnode,'target_image', warpmean,'reference_image')
    register.connect(inputnode, 'mean_image', warpmean, 'input_image')
    register.connect(merge, 'out', warpmean, 'transforms')

    """
    Assign all the output files
    """

    register.connect(reg, 'warped_image', outputnode, 'anat2target')
    register.connect(aparcxfm, 'transformed_file',
                     outputnode, 'aparc')
    register.connect(bbregister, 'out_fsl_file',
                     outputnode, 'func2anat_transform')
    register.connect(bbregister, 'out_reg_file',
                     outputnode, 'out_reg_file')
    register.connect(bbregister, 'min_cost_file',
                     outputnode, 'min_cost_file')
    register.connect(mean2anat_mask, 'mask_file',
                     outputnode, 'mean2anat_mask')
    register.connect(reg, 'composite_transform',
                     outputnode, 'anat2target_transform')
    register.connect(merge, 'out', outputnode, 'transforms')
    register.connect(maskxfm, "transformed_file", outputnode, "mask_file")

    return register


"""
Get info for a given subject
"""

def get_subjectinfo(subject_id, base_dir, task_id, model_id, session_id=None):
    """Get info for a given subject

    Parameters
    ----------
    subject_id : string
        Subject identifier (e.g., sub001)
    base_dir : string
        Path to base directory of the dataset
    task_id : int
        Which task to process
    model_id : int
        Which model to process

    Returns
    -------
    run_ids : list of ints
        Run numbers
    conds : list of str
        Condition names
    TR : float
        Repetition time
    """
    from glob import glob
    import os
    import numpy as np
    import re
    
    condition_info = []
    cond_file = os.path.join(base_dir, 'code', 'model', 'model%03d' % model_id,
                                 'condition_key.txt') 
    with open(cond_file, 'rt') as fp:
        for line in fp:
            info = line.strip().split()
            condition_info.append([info[0], info[1], ' '.join(info[2:])])
    if len(condition_info) == 0:
        raise ValueError('No condition info found in %s' % cond_file)
    taskinfo = np.array(condition_info)
    n_tasks = np.unique(taskinfo[:, 0])
    conds = []
    run_ids = []
    if task_id > len(n_tasks):
        raise ValueError('Task id %s does not exist' % task_id)
    for idx,task in enumerate(n_tasks):
        taskidx = np.where(taskinfo[:, 0] == '%s'%(task))
        conds.append([condition.replace(' ', '_') for condition
                      in taskinfo[taskidx[0], 2]]) # if 'junk' not in condition])
        if session_id:
            files = sorted(glob(os.path.join(base_dir,
                                             subject_id,
                                             session_id,
                                             'func',
                                             '*%s*.nii.gz'%(task))))
        else:
            files = sorted(glob(os.path.join(base_dir,
                                             subject_id,
                                             'func',
                                             '*%s*.nii.gz'%(task))))
            
        runs = [int(re.search('(?<=run-)\d+',os.path.basename(val)).group(0)) for val in files]
        run_ids.insert(idx, runs)
    # TR should be same across runs
    if session_id:
        json_info = glob(os.path.join(base_dir, subject_id, session_id, 
                                      'func','*%s*.json'%(n_tasks[task_id-1])))[0]
    else:    
        json_info = glob(os.path.join(base_dir, subject_id, 'func',
                                     '*%s*.json'%(n_tasks[task_id-1])))[0]
    if os.path.exists(json_info):
        import json
        with open(json_info, 'rt') as fp:
            data = json.load(fp)
            TR = data['RepetitionTime']
    else:
        task_scan_key = os.path.join(base_dir, 'code', 'scan_key.txt')
        if os.path.exists(task_scan_key):
            TR = np.genfromtxt(task_scan_key)[1]
        else:
            TR = np.genfromtxt(os.path.join(base_dir, 'scan_key.txt'))[1]
    return run_ids[task_id - 1], conds[task_id - 1], TR

def extract_subrois(timeseries_file, label_file, indices):
    """Extract voxel time courses for each subcortical roi index

    Parameters
    ----------

    timeseries_file: a 4D Nifti file
    label_file: a 3D file containing rois in the same space/size of the 4D file
    indices: a list of indices for ROIs to extract.

    Returns
    -------
    out_file: a text file containing time courses for each voxel of each roi
        The first four columns are: freesurfer index, i, j, k positions in the
        label file
    """
    img = nb.load(timeseries_file)
    data = img.get_data()
    roiimg = nb.load(label_file)
    rois = roiimg.get_data()
    prefix = split_filename(timeseries_file)[1]
    out_ts_file = os.path.join(os.getcwd(), '%s_subcortical_ts.txt' % prefix)
    with open(out_ts_file, 'wt') as fp:
        for fsindex in indices:
            ijk = np.nonzero(rois == fsindex)
            ts = data[ijk]
            for i0, row in enumerate(ts):
                fp.write('%d,%d,%d,%d,' % (fsindex, ijk[0][i0],
                                           ijk[1][i0], ijk[2][i0]) +
                         ','.join(['%.10f' % val for val in row]) + '\n')
    return out_ts_file


def combine_hemi(left, right):
    """Combine left and right hemisphere time series into a single text file
    """
    lh_data = nb.load(left).get_data()
    rh_data = nb.load(right).get_data()

    indices = np.vstack((1000000 + np.arange(0, lh_data.shape[0])[:, None],
                         2000000 + np.arange(0, rh_data.shape[0])[:, None]))
    all_data = np.hstack((indices, np.vstack((lh_data.squeeze(),
                                              rh_data.squeeze()))))
    filename = left.split('.')[1] + '_combined.txt'
    np.savetxt(filename, all_data,
               fmt=','.join(['%d'] + ['%.10f'] * (all_data.shape[1] - 1)))
    return os.path.abspath(filename)

def get_taskname(base_dir, task_id):
    import os
    task_key = os.path.join(base_dir, 'code', 'task_key.txt')
    if not os.path.exists(task_key):
        return
    
    with open(task_key, 'rt') as fp:
        for line in fp:
            info = line.strip().split()
            if 'task%03d'%(task_id) in info:
                return info[1]
"""
Analyzes an open fmri dataset
"""

def analyze_openfmri_dataset(data_dir, subject=None, model_id=None,
                             task_id=None, output_dir=None, subj_prefix='*',
                             hpcutoff=120., use_derivatives=True,
                             fwhm=6.0, subjects_dir=None, target=None,
                             surf_fwhm=None, 
                             target_subject=['fsaverage3', 'fsaverage4']):
    """Analyzes an open fmri dataset

    Parameters
    ----------

    data_dir : str
        Path to the base data directory

    work_dir : str
        Nipype working directory (defaults to cwd)
    """

    """
    Load nipype workflows
    """

    preproc = create_featreg_preproc(whichvol='first')
    modelfit = create_modelfit_workflow()
    fixed_fx = create_fixed_effects_flow()
    if subjects_dir:
        registration = create_fs_reg_workflow()
    else:
        registration = create_reg_workflow()

    """
    Remove the plotting connection so that plot iterables don't propagate
    to the model stage
    """

    preproc.disconnect(preproc.get_node('plot_motion'), 'out_file',
                       preproc.get_node('outputspec'), 'motion_plots')

    """
    Set up openfmri data specific components
    """

    subjects = sorted([path.split(os.path.sep)[-1] for path in
                       glob(os.path.join(data_dir, subj_prefix))])

    infosource = pe.Node(niu.IdentityInterface(fields=['subject_id',
                                                       'model_id',
                                                       'task_id']),
                         name='infosource')
    if len(subject) == 0:
        infosource.iterables = [('subject_id', subjects),
                                ('model_id', [model_id]),
                                ('task_id', task_id)]
    else:
        infosource.iterables = [('subject_id',
                                 [subjects[subjects.index(subj)] for subj in subject]),
                                ('model_id', [model_id]),
                                ('task_id', task_id)]

    subjinfo = pe.Node(niu.Function(input_names=['subject_id', 'base_dir',
                                                 'task_id', 'model_id', 'session_id'],
                                    output_names=['run_id', 'conds', 'TR'],
                                    function=get_subjectinfo),
                       name='subjectinfo')
    subjinfo.inputs.session_id = None
    subjinfo.inputs.base_dir = data_dir

    """
    Get task name (BIDS)
    """
    taskname = pe.Node(niu.Function(input_names=['base_dir', 'task_id'],
                                    output_names=['task_name'],
                                    function=get_taskname),
                       name='taskname')
    taskname.inputs.base_dir = data_dir

    """
    Return data components as anat, bold and behav
    """
    contrast_file = os.path.join(data_dir, 'code', 'model', 'model%03d' % model_id,
                                 'task_contrasts.txt')
    
    has_contrast = os.path.exists(contrast_file)
    if has_contrast:
        datasource = pe.Node(nio.DataGrabber(infields=['subject_id', 'run_id',
                                                   'task_id', 'model_id', 'task_name'],
                                         outfields=['anat', 'bold', 'behav',
                                                    'contrasts']),
                         name='datasource')
    else:
        datasource = pe.Node(nio.DataGrabber(infields=['subject_id', 'run_id',
                                                   'task_id', 'model_id', 'task_name'],
                                         outfields=['anat', 'bold', 'behav']),
                         name='datasource')
    datasource.inputs.base_directory = data_dir

    datasource.inputs.template = '*'
########## 6/23/16 replace behav with events.tsv
    if has_contrast:
        datasource.inputs.field_template = {'anat': '%s/anat/*T1w.nii.gz',
                                            'bold': '%s/func/*task-%s_*bold.nii.gz',
                                            'behav': ('code/model/model%03d/onsets/%s/task%03d_'
                                                      'run%03d/cond*.txt'), 
                                            'contrasts': ('code/model/model%03d/'
                                                          'task_contrasts.txt')}
        datasource.inputs.template_args = {'anat': [['subject_id']],
                                       'bold': [['subject_id', 'task_name']],
                                       'behav': [['model_id', 'subject_id',
                                                  'task_id', 'run_id']],
                                       'contrasts': [['model_id']]}
    else:
        datasource.inputs.field_template = {'anat': '%s/anat/*T1w.nii.gz',
                                            'bold': '%s/func/*task-%s_*bold.nii.gz',
                                            'behav': ('code/model/model%03d/onsets/%s/task%03d_'
                                                      'run%03d/cond*.txt')}
        datasource.inputs.template_args = {'anat': [['subject_id']],
                                       'bold': [['subject_id', 'task_name']],
                                       'behav': [['model_id', 'subject_id',
                                                  'task_id', 'run_id']]}

    datasource.inputs.sort_filelist = True

    """
    Create meta workflow
    """

    wf = pe.Workflow(name='openfmri')
    wf.connect(infosource, 'subject_id', subjinfo, 'subject_id')
    wf.connect(infosource, 'model_id', subjinfo, 'model_id')
    wf.connect(infosource, 'task_id', subjinfo, 'task_id')
    wf.connect(infosource, 'task_id', taskname, 'task_id')
    wf.connect(taskname, 'task_name', datasource, 'task_name')
    wf.connect(infosource, 'subject_id', datasource, 'subject_id')
    wf.connect(infosource, 'model_id', datasource, 'model_id')
    wf.connect(infosource, 'task_id', datasource, 'task_id')
    wf.connect(subjinfo, 'run_id', datasource, 'run_id')
    wf.connect([(datasource, preproc, [('bold', 'inputspec.func')]),
                ])

    def get_highpass(TR, hpcutoff):
        return hpcutoff / (2 * TR)
    gethighpass = pe.Node(niu.Function(input_names=['TR', 'hpcutoff'],
                                       output_names=['highpass'],
                                       function=get_highpass),
                          name='gethighpass')
    wf.connect(subjinfo, 'TR', gethighpass, 'TR')
    wf.connect(gethighpass, 'highpass', preproc, 'inputspec.highpass')

    """
    Setup a basic set of contrasts, a t-test per condition
    """

    def get_contrasts(contrast_file, task_name, conds):
        import numpy as np
        import os
        contrast_def = []
        if os.path.exists(contrast_file):
            with open(contrast_file, 'rt') as fp:
                contrast_def.extend([np.array(row.split()) for row in fp.readlines() if row.strip()])
        contrasts = []
        for row in contrast_def:
            if row[0] != 'task-%s' % task_name:
                continue
            con = [row[1], 'T', ['cond%03d' % (i + 1)  for i in range(len(conds))],
                   row[2:].astype(float).tolist()]
            contrasts.append(con)
        # add auto contrasts for each column
        for i, cond in enumerate(conds):
            con = [cond, 'T', ['cond%03d' % (i + 1)], [1]]
            contrasts.append(con)
        return contrasts

    contrastgen = pe.Node(niu.Function(input_names=['contrast_file',
                                                    'task_name', 'conds'],
                                       output_names=['contrasts'],
                                       function=get_contrasts),
                          name='contrastgen')

    art = pe.MapNode(interface=ra.ArtifactDetect(use_differences=[True, False],
                                                 use_norm=True,
                                                 norm_threshold=1,
                                                 zintensity_threshold=3,
                                                 parameter_source='FSL',
                                                 mask_type='file'),
                     iterfield=['realigned_files', 'realignment_parameters',
                                'mask_file'],
                     name="art")

    modelspec = pe.Node(interface=model.SpecifyModel(),
                           name="modelspec")
    modelspec.inputs.input_units = 'secs'

    if use_spm_model:
        modelspec_spm = pe.Node(interface=model.SpecifySPMModel(),
                            name="modelspec_spm")
        modelspec_spm.inputs.input_units = 'secs'
        modelspec_spm.inputs.high_pass_filter_cutoff = hpcutoff

    def check_behav_list(behav, run_id, conds):
        from nipype.external import six
        import numpy as np
        num_conds = len(conds)
        if isinstance(behav, six.string_types):
            behav = [behav]
        behav_array = np.array(behav).flatten()
        num_elements = behav_array.shape[0]
        return behav_array.reshape(num_elements/num_conds, num_conds).tolist()

    reshape_behav = pe.Node(niu.Function(input_names=['behav', 'run_id', 'conds'],
                                       output_names=['behav'],
                                       function=check_behav_list),
                          name='reshape_behav')

    wf.connect(subjinfo, 'TR', modelspec, 'time_repetition')
    wf.connect(datasource, 'behav', reshape_behav, 'behav')
    wf.connect(subjinfo, 'run_id', reshape_behav, 'run_id')
    wf.connect(subjinfo, 'conds', reshape_behav, 'conds')
    wf.connect(reshape_behav, 'behav', modelspec, 'event_files')

    if use_spm_model:
        wf.connect(subjinfo, 'TR', modelspec_spm, 'time_repetition')
        wf.connect(reshape_behav, 'behav', modelspec_spm, 'event_files')

    wf.connect(subjinfo, 'TR', modelfit, 'inputspec.interscan_interval')
    wf.connect(subjinfo, 'conds', contrastgen, 'conds')
    if has_contrast:
        wf.connect(datasource, 'contrasts', contrastgen, 'contrast_file')
    else:
        contrastgen.inputs.contrast_file = ''
    wf.connect(taskname, 'task_name', contrastgen, 'task_name')
    wf.connect(contrastgen, 'contrasts', modelfit, 'inputspec.contrasts')

    maskfunc3 = preproc.get_node('maskfunc3')
    maskfunc3.inputs.output_type = 'NIFTI'
    
    if use_spm_smooth:
        smooth = pe.Node(spm.Smooth(), name='smooth')
        smooth.inputs.fwhm = fwhm

        realign = preproc.get_node('realign')
        realign.inputs.output_type = 'NIFTI'
        preproc.connect(realign, 'out_file', smooth, 'in_files')
        #maskfunc2 = preproc.get_node('maskfunc2')
        #maskfunc2.inputs.output_type = 'NIFTI'
        #preproc.connect(maskfunc2, 'out_file', smooth, 'in_files')

        susansmooth = preproc.get_node('susan_smooth')
        preproc.disconnect(susansmooth, 'outputnode.smoothed_files', maskfunc3, 'in_file')
        preproc.connect(smooth, 'smoothed_files', maskfunc3, 'in_file')
        
    wf.connect([(preproc, art, [('outputspec.motion_parameters',
                                 'realignment_parameters'),
                                ('outputspec.realigned_files',
                                 'realigned_files'),
                                ('outputspec.mask', 'mask_file')]),
                (preproc, modelspec, [('outputspec.highpassed_files',
                                       'functional_runs'),
                                      ('outputspec.motion_parameters',
                                       'realignment_parameters')]),
                (art, modelspec, [('outlier_files', 'outlier_files')]),
                (modelspec, modelfit, [('session_info',
                                        'inputspec.session_info')]),
                (preproc, modelfit, [('outputspec.highpassed_files',
                                      'inputspec.functional_data')])
                ])

    pickfirst = lambda x: x[0]

    if use_spm_model:
        wf.connect([(preproc, modelspec_spm, [('outputspec.motion_parameters',
                                           'realignment_parameters')]),
                    (art, modelspec_spm, [('outlier_files', 'outlier_files')])
                    ])
        dilatemask = preproc.get_node('dilatemask')
        dilatemask.inputs.output_type = 'NIFTI'
        if use_spm_smooth:
            wf.connect(preproc, 'smooth.smoothed_files', modelspec_spm, 'functional_runs')
        else:
            wf.connect(preproc, 'maskfunc3.out_file', modelspec_spm, 'functional_runs')

        """Generate a first level SPM.mat file for analysis
        :class:`nipype.interfaces.spm.Level1Design`.
        """

        level1design = pe.Node(interface=spm.Level1Design(), name= "level1design")
        level1design.inputs.timing_units = modelspec_spm.inputs.output_units
        if subjects_dir and use_spm_smooth:
            #wf.connect(registration, 'outputspec.mask_file', level1design, 'mask_image')
            pass
        else:
            wf.connect(dilatemask, ('out_file', pickfirst), level1design, 'mask_image')
        wf.connect(subjinfo, 'TR', level1design, 'interscan_interval')
        level1design.inputs.bases = {'hrf': {'derivs': [int(use_derivatives), 0]}}
        level1design.inputs.model_serial_correlations = 'AR(1)'
        
        """Use :class:`nipype.interfaces.spm.EstimateModel` to determine the
        parameters of the model.
        """

        level1estimate = pe.Node(interface=spm.EstimateModel(), name="level1estimate")
        level1estimate.inputs.estimation_method = {'Classical' : 1}

    
        """Use :class:`nipype.interfaces.spm.EstimateContrast` to estimate the
        first level contrasts specified in a few steps above.
        """
    
        contrastestimate = pe.Node(interface = spm.EstimateContrast(), name="contrastestimate")
        wf.connect(contrastgen, 'contrasts', contrastestimate, 'contrasts')

        wf.connect([(modelspec_spm, level1design, [('session_info', 'session_info')]),
                    (level1design, level1estimate, [('spm_mat_file', 'spm_mat_file')]),
                    (level1estimate, contrastestimate, [('spm_mat_file', 'spm_mat_file'),
                                                        ('beta_images', 'beta_images'),
                                                        ('residual_image', 'residual_image')]),
                    ])

    # Comute TSNR on realigned data regressing polynomials upto order 2
    tsnr = MapNode(TSNR(regress_poly=2), iterfield=['in_file'], name='tsnr')
    wf.connect(preproc, "outputspec.realigned_files", tsnr, "in_file")

    # Compute the median image across runs
    calc_median = Node(Function(input_names=['in_files'],
                                output_names=['median_file'],
                                function=median,
                                imports=imports),
                       name='median')
    wf.connect(tsnr, 'detrended_file', calc_median, 'in_files')

    """
    Reorder the copes so that now it combines across runs
    """

    def sort_copes(copes, varcopes, contrasts):
        import numpy as np
        if not isinstance(copes, list):
            copes = [copes]
            varcopes = [varcopes]
        num_copes = len(contrasts)
        n_runs = len(copes)
        all_copes = np.array(copes).flatten()
        all_varcopes = np.array(varcopes).flatten()
        outcopes = all_copes.reshape(len(all_copes)/num_copes, num_copes).T.tolist()
        outvarcopes = all_varcopes.reshape(len(all_varcopes)/num_copes, num_copes).T.tolist()
        return outcopes, outvarcopes, n_runs

    cope_sorter = pe.Node(niu.Function(input_names=['copes', 'varcopes',
                                                    'contrasts'],
                                       output_names=['copes', 'varcopes',
                                                     'n_runs'],
                                       function=sort_copes),
                          name='cope_sorter')

    wf.connect(contrastgen, 'contrasts', cope_sorter, 'contrasts')
    wf.connect([(preproc, fixed_fx, [(('outputspec.mask', pickfirst),
                                      'flameo.mask_file')]),
                (modelfit, cope_sorter, [('outputspec.copes', 'copes')]),
                (modelfit, cope_sorter, [('outputspec.varcopes', 'varcopes')]),
                (cope_sorter, fixed_fx, [('copes', 'inputspec.copes'),
                                         ('varcopes', 'inputspec.varcopes'),
                                         ('n_runs', 'l2model.num_copes')]),
                (modelfit, fixed_fx, [('outputspec.dof_file',
                                        'inputspec.dof_files'),
                                      ])
                ])

    wf.connect(calc_median, 'median_file', registration, 'inputspec.mean_image')
    if subjects_dir:
        wf.connect(infosource, 'subject_id', registration, 'inputspec.subject_id')
        registration.inputs.inputspec.subjects_dir = subjects_dir
        computed_target = fsl.Info.standard_image('MNI152_T1_2mm_brain.nii.gz')
        if target:
            computed_target = target
        registration.inputs.inputspec.target_image = computed_target
    else:
        wf.connect(datasource, 'anat', registration, 'inputspec.anatomical_image')
        registration.inputs.inputspec.target_image = fsl.Info.standard_image('MNI152_T1_2mm.nii.gz')
        registration.inputs.inputspec.target_image_brain = fsl.Info.standard_image('MNI152_T1_2mm_brain.nii.gz')
        registration.inputs.inputspec.config_file = 'T1_2_MNI152_2mm'

    def merge_files(copes, varcopes, zstats, spm_con=None, spm_tstats=None):
        out_files = []
        splits = []
        out_files.extend(copes)
        splits.append(len(copes))
        out_files.extend(varcopes)
        splits.append(len(varcopes))
        out_files.extend(zstats)
        splits.append(len(zstats))
        if spm_con is not None:
            out_files.extend(spm_con)
            splits.append(len(spm_con))
            out_files.extend(spm_tstats)
            splits.append(len(spm_tstats))
        return out_files, splits
    
    mergefunc = pe.Node(niu.Function(input_names=['copes', 'varcopes',
                                                  'zstats', 'spm_con', 'spm_tstats'],
                                   output_names=['out_files', 'splits'],
                                   function=merge_files),
                      name='merge_files')
    wf.connect([(fixed_fx.get_node('outputspec'), mergefunc,
                                 [('copes', 'copes'),
                                  ('varcopes', 'varcopes'),
                                  ('zstats', 'zstats'),
                                  ])])
    
    if use_spm_model:
        wf.connect([(contrastestimate, mergefunc,
                     [('con_images', 'spm_con'),
                      ('spmT_images', 'spm_tstats')])
                    ])

    if subjects_dir:
        """
        Transform the remaining images. First to anatomical and then to target
        """

        warpall = pe.MapNode(ants.ApplyTransforms(),
                             iterfield=['input_image'],
                             name='warpall')
        warpall.inputs.input_image_type = 0
        warpall.inputs.interpolation = 'Linear'
        warpall.inputs.invert_transform_flags = [False, False]
        warpall.inputs.terminal_output = 'file'
        warpall.inputs.num_threads = 2
        warpall.plugin_args = {'sbatch_args': '--mem=6G -c 2'}
        warpall.inputs.reference_image = computed_target
        wf.connect(mergefunc, 'out_files', warpall, 'input_image')
        wf.connect(registration, 'outputspec.transforms', warpall, 'transforms')
    else:
        wf.connect(mergefunc, 'out_files', registration, 'inputspec.source_files')
        
    def split_files(in_files, splits):
        copes = in_files[:splits[0]]
        varcopes = in_files[splits[0]:(splits[0] + splits[1])]
        zstats = in_files[(splits[0] + splits[1]):sum(splits[:3])]
        spm_con = None
        spm_tstats = None
        if len(splits) > 3:
            spm_con = in_files[sum(splits[:3]):sum(splits[:4])]
            spm_tstats = in_files[sum(splits[:4]):sum(splits[:5])]
        return copes, varcopes, zstats, spm_con, spm_tstats

    splitfunc = pe.Node(niu.Function(input_names=['in_files', 'splits'],
                                     output_names=['copes', 'varcopes',
                                                   'zstats', 'spm_con',
                                                   'spm_tstats'],
                                     function=split_files),
                      name='split_files')
    wf.connect(mergefunc, 'splits', splitfunc, 'splits')
    if subjects_dir:
        wf.connect(warpall, 'output_image',
                   splitfunc, 'in_files')
    else:
        wf.connect(registration, 'outputspec.transformed_files',
                   splitfunc, 'in_files')

    if use_spm_model:
        genvarcope = pe.MapNode(interface=fsl.maths.MultiImageMaths(nan2zeros=True,
                                                                    op_string='-div %s -mul 10000'),
                                iterfield=['in_file', 'operand_files'],
                          name='genvarcope')
        wf.connect([(splitfunc, genvarcope,
                     [('spm_con', 'in_file'),
                      ('spm_tstats', 'operand_files')])
                    ])

    if subjects_dir:
        get_roi_mean = pe.MapNode(fs.SegStats(default_color_table=True),
                                  iterfield=['in_file'], name='get_aparc_means')
        get_roi_mean.inputs.avgwf_txt_file = True
        wf.connect(fixed_fx.get_node('outputspec'), 'copes', get_roi_mean, 'in_file')
        wf.connect(registration, 'outputspec.aparc', get_roi_mean, 'segmentation_file')
        if use_spm_model:
            get_roi_mean2 = get_roi_mean.clone('get_spm_aparc_means')
            wf.connect(contrastestimate, 'con_images', get_roi_mean2, 'in_file')
            wf.connect(registration, 'outputspec.aparc', get_roi_mean2, 'segmentation_file')
        get_roi_tsnr = pe.MapNode(fs.SegStats(default_color_table=True),
                                  iterfield=['in_file'], name='get_aparc_tsnr')
        get_roi_tsnr.inputs.avgwf_txt_file = True
        wf.connect(tsnr, 'tsnr_file', get_roi_tsnr, 'in_file')
        wf.connect(registration, 'outputspec.aparc', get_roi_tsnr, 'segmentation_file')

        # Sample the average time series in aparc ROIs
        # from rsfmri_vol_surface_preprocessing_nipy.py
        sampleaparc = MapNode(freesurfer.SegStats(default_color_table=True),
                              iterfield=['in_file'],
                              name='aparc_ts')
        sampleaparc.inputs.segment_id = ([8] + range(10, 14) + [17, 18, 26, 47] +
                                         range(49, 55) + [58] + range(1001, 1036) +
                                         range(2001, 2036))
        sampleaparc.inputs.avgwf_txt_file = True

        wf.connect(registration, 'outputspec.aparc', sampleaparc, 'segmentation_file')
        wf.connect(preproc, 'outputspec.realigned_files', sampleaparc, 'in_file')

        target = Node(IdentityInterface(fields=['target_subject']), name='target')
        target.iterables = ('target_subject', filename_to_list(target_subject))

        samplerlh = MapNode(freesurfer.SampleToSurface(),
                            iterfield=['source_file'],
                            name='sampler_lh')
        samplerlh.inputs.sampling_method = "average"
        samplerlh.inputs.sampling_range = (0.1, 0.9, 0.1)
        samplerlh.inputs.sampling_units = "frac"
        samplerlh.inputs.interp_method = "trilinear"
        samplerlh.inputs.smooth_surf = surf_fwhm
        samplerlh.inputs.out_type = 'niigz'
        samplerlh.inputs.subjects_dir = subjects_dir

        samplerrh = samplerlh.clone('sampler_rh')

        samplerlh.inputs.hemi = 'lh'
        wf.connect(preproc, 'outputspec.realigned_files', samplerlh, 'source_file')
        wf.connect(registration, 'outputspec.out_reg_file', samplerlh, 'reg_file')
        wf.connect(target, 'target_subject', samplerlh, 'target_subject')
        
        samplerrh.set_input('hemi', 'rh')
        wf.connect(preproc, 'outputspec.realigned_files', samplerrh, 'source_file')
        wf.connect(registration, 'outputspec.out_reg_file', samplerrh, 'reg_file')
        wf.connect(target, 'target_subject', samplerrh, 'target_subject')

        # Combine left and right hemisphere to text file
        combiner = MapNode(Function(input_names=['left', 'right'],
                                    output_names=['out_file'],
                                    function=combine_hemi,
                                    imports=imports),
                           iterfield=['left', 'right'],
                           name="combiner")
        wf.connect(samplerlh, 'out_file', combiner, 'left')
        wf.connect(samplerrh, 'out_file', combiner, 'right')

        # Sample the time series file for each subcortical roi
        ts2txt = MapNode(Function(input_names=['timeseries_file', 'label_file',
                                               'indices'],
                                  output_names=['out_file'],
                                  function=extract_subrois,
                                  imports=imports),
                         iterfield=['timeseries_file'],
                         name='getsubcortts')
        ts2txt.inputs.indices = [8] + list(range(10, 14)) + [17, 18, 26, 47] +\
            list(range(49, 55)) + [58]
        wf.connect(registration, 'outputspec.aparc', ts2txt, 'label_file')
        wf.connect(preproc, 'outputspec.realigned_files', ts2txt, 'timeseries_file')


    """
    Connect to a datasink
    """

    def get_subs(subject_id, conds, run_id, model_id, task_id):
        subs = [('_subject_id_%s_' % subject_id, ''),
                ('_target_subject_', ''),
                ]
        subs.append(('_model_id_%d' % model_id, 'model%03d' %model_id))
        subs.append(('task_id_%d/' % task_id, '/task%03d_' % task_id))
        subs.append(('bold_dtype_mcf_mask_smooth_mask_gms_tempfilt_mean_warp',
        'mean'))
        subs.append(('bold_dtype_mcf_mask_smooth_mask_gms_tempfilt_mean_flirt',
        'affine'))

        for i in range(len(conds)):
            subs.append(('_flameo%d/cope1.' % i, 'cope%02d.' % (i + 1)))
            subs.append(('_flameo%d/varcope1.' % i, 'varcope%02d.' % (i + 1)))
            subs.append(('_flameo%d/zstat1.' % i, 'zstat%02d.' % (i + 1)))
            subs.append(('_flameo%d/tstat1.' % i, 'tstat%02d.' % (i + 1)))
            subs.append(('_flameo%d/res4d.' % i, 'res4d%02d.' % (i + 1)))
            subs.append(('_warpall%d/cope1_warp.' % i,
                         'cope%02d.' % (i + 1)))
            subs.append(('_warpall%d/varcope1_warp.' % (len(conds) + i),
                         'varcope%02d.' % (i + 1)))
            subs.append(('_warpall%d/zstat1_warp.' % (2 * len(conds) + i),
                         'zstat%02d.' % (i + 1)))
            subs.append(('_warpall%d/cope1_trans.' % i,
                         'cope%02d.' % (i + 1)))
            subs.append(('_warpall%d/varcope1_trans.' % (len(conds) + i),
                         'varcope%02d.' % (i + 1)))
            subs.append(('_warpall%d/zstat1_trans.' % (2 * len(conds) + i),
                         'zstat%02d.' % (i + 1)))
            subs.append(('_warpall%d/con_%04d_trans.' % (3 * len(conds) + i, i + 1), 
                         'cope%02d.' % (i + 1)))
            subs.append(('_genvarcope%d/con_%04d_trans_maths.' % (i, i + 1), 
                         'varcope%02d.' % (i + 1)))
            subs.append(('_warpall%d/spmT_%04d_trans.' % (4 * len(conds) + i, i + 1), 
                         'tstat%02d.' % (i + 1)))
            subs.append(('_get_aparc_means%d/' % i, 'cope%02d_' % (i + 1)))
            subs.append(('_get_spm_aparc_means%d/' % i, 'cope%02d_' % (i + 1)))

        for i, run_num in enumerate(run_id):
            subs.append(('__get_aparc_tsnr%d/' % i, '/run%02d_' % run_num))
            subs.append(('__art%d/' % i, '/run%02d_' % run_num))
            subs.append(('__dilatemask%d/' % i, '/run%02d_' % run_num))
            subs.append(('__realign%d/' % i, '/run%02d_' % run_num))
            subs.append(('__modelgen%d/' % i, '/run%02d_' % run_num))
            subs.append(('_getsubcortts%d/' % i, '/run%02d_' % run_num))
            subs.append(('_combiner%d/' % i, '/run%02d_' % run_num))
        subs.append(('/model%03d/task%03d/' % (model_id, task_id), '/'))
        subs.append(('/model%03d/task%03d_' % (model_id, task_id), '/'))
        subs.append(('_bold_dtype_mcf_bet_thresh_dil', '_mask'))
        subs.append(('_bold_dtype_mcf_combined', '_bold_surface_timeseries'))
        subs.append(('_bold_dtype_mcf_subcortical_ts', '_bold_subcortical_timeseries'))
        subs.append(('_output_warped_image', '_anat2target'))
        subs.append(('median_flirt_brain_mask', 'median_brain_mask'))
        subs.append(('median_bbreg_brain_mask', 'median_brain_mask'))
        subs.append(('/model%02d/' % model_id, '/model%03d/' % model_id))
        return subs

    subsgen = pe.Node(niu.Function(input_names=['subject_id', 'conds', 'run_id',
                                                'model_id', 'task_id'],
                                   output_names=['substitutions'],
                                   function=get_subs),
                      name='subsgen')
    wf.connect(subjinfo, 'run_id', subsgen, 'run_id')

    datasink = pe.Node(interface=nio.DataSink(),
                       name="datasink")
    wf.connect(infosource, 'subject_id', datasink, 'container')
    wf.connect(infosource, 'subject_id', subsgen, 'subject_id')
    wf.connect(infosource, 'model_id', subsgen, 'model_id')
    wf.connect(infosource, 'task_id', subsgen, 'task_id')
    wf.connect(contrastgen, 'contrasts', subsgen, 'conds')
    wf.connect(subsgen, 'substitutions', datasink, 'substitutions')
    if use_spm_model:
        wf.connect([(contrastestimate, datasink,
                     [('con_images', 'copes.spm'),
                      ('spmT_images', 'tstats.spm')])
                    ])
    wf.connect([(fixed_fx.get_node('outputspec'), datasink,
                                 [('res4d', 'res4d'),
                                  ('copes', 'copes'),
                                  ('varcopes', 'varcopes'),
                                  ('zstats', 'zstats'),
                                  ('tstats', 'tstats')])
                                 ])
    wf.connect([(modelfit.get_node('modelgen'), datasink,
                                 [('design_cov', 'qa.model'),
                                  ('design_image', 'qa.model.@matrix_image'),
                                  ('design_file', 'qa.model.@matrix'),
                                 ])])
    if use_spm_model:
        wf.connect(level1estimate, 'mask_image', datasink, 'qa.model.spm_mask')
    wf.connect([(preproc, datasink, [('outputspec.motion_parameters',
                                      'qa.motion'),
                                     ('outputspec.motion_plots',
                                      'qa.motion.plots'),
                                     ('outputspec.mask', 'qa.mask')])])
    wf.connect(registration, 'outputspec.mean2anat_mask', datasink, 'qa.mask.mean2anat')
    wf.connect(art, 'norm_files', datasink, 'qa.art.@norm')
    wf.connect(art, 'intensity_files', datasink, 'qa.art.@intensity')
    wf.connect(art, 'outlier_files', datasink, 'qa.art.@outlier_files')
    wf.connect(registration, 'outputspec.anat2target', datasink, 'qa.anat2target')
    wf.connect(tsnr, 'tsnr_file', datasink, 'qa.tsnr.@map')
    if subjects_dir:
        wf.connect(registration, 'outputspec.min_cost_file', datasink, 'qa.mincost')
        wf.connect(registration, 'outputspec.mask_file', datasink, 'qa.mask.@fsmask')
        wf.connect([(get_roi_tsnr, datasink, [('avgwf_txt_file', 'qa.tsnr'),
                                              ('summary_file', 'qa.tsnr.@summary')])])
        wf.connect([(get_roi_mean, datasink, [('avgwf_txt_file', 'copes.roi'),
                                              ('summary_file', 'copes.roi.@summary')])])
        wf.connect([(get_roi_mean2, datasink, [('avgwf_txt_file', 'copes.roi.spm'),
                                              ('summary_file', 'copes.roi.spm.@summary')])])
        wf.connect(sampleaparc, 'summary_file', datasink, 'timeseries.aparc.@summary')
        wf.connect(sampleaparc, 'avgwf_txt_file', datasink, 'timeseries.aparc')
        wf.connect(ts2txt, 'out_file',
                   datasink, 'timeseries.grayo.@subcortical')
    wf.connect([(splitfunc, datasink,
                 [('copes', 'copes.mni'),
                  ('varcopes', 'varcopes.mni'),
                  ('zstats', 'zstats.mni'),
                  ])])
    if use_spm_model:
        wf.connect([(splitfunc, datasink,
                     [('spm_con', 'copes.spm.mni'),
                      ('spm_tstats', 'tstats.spm.mni'),
                      ]),
                    (genvarcope, datasink, [('out_file', 'varcopes.spm.mni')]),
                    ])

    wf.connect(calc_median, 'median_file', datasink, 'mean')
    wf.connect(registration, 'outputspec.transformed_mean', datasink, 'mean.mni')
    wf.connect(registration, 'outputspec.func2anat_transform', datasink, 'xfm.mean2anat')
    wf.connect(registration, 'outputspec.anat2target_transform', datasink, 'xfm.anat2target')

    if subjects_dir:
        datasink2 = Node(interface=DataSink(), name="datasink2")
        wf.connect(infosource, 'subject_id', datasink2, 'container')
        wf.connect(subsgen, 'substitutions', datasink2, 'substitutions')
        wf.connect(combiner, 'out_file',
                   datasink2, 'timeseries.grayo.@surface')

    """
    Set processing parameters
    """

    preproc.inputs.inputspec.fwhm = fwhm
    gethighpass.inputs.hpcutoff = hpcutoff
    modelspec.inputs.high_pass_filter_cutoff = hpcutoff
    modelfit.inputs.inputspec.bases = {'dgamma': {'derivs': use_derivatives}}
    modelfit.inputs.inputspec.model_serial_correlations = True
    modelfit.inputs.inputspec.film_threshold = 1000

    datasink.inputs.base_directory = output_dir
    if subjects_dir:
        datasink2.inputs.base_directory = output_dir
    return wf

"""
The following functions run the whole workflow.
"""

if __name__ == '__main__':
    import argparse
    defstr = ' (default %(default)s)'
    parser = argparse.ArgumentParser(prog='fmri_openfmri.py',
                                     description=__doc__)
    parser.add_argument('-d', '--datasetdir', required=True)
    parser.add_argument('-s', '--subject', default=[],
                        nargs='+', type=str,
                        help="Subject name (e.g. 'sub001')")
    parser.add_argument('-m', '--model', default=1,
                        help="Model index" + defstr)
    parser.add_argument('-x', '--subjectprefix', default='sub*',
                        help="Subject prefix" + defstr)
    parser.add_argument('-t', '--task', default=1, #nargs='+',
                        type=int, help="Task index" + defstr)
    parser.add_argument('--hpfilter', default=120.,
                        type=float, help="High pass filter cutoff (in secs)" + defstr)
    parser.add_argument('--fwhm', default=6.,
                        type=float, help="Spatial FWHM" + defstr)
    parser.add_argument('--derivatives', action="store_true",
                        help="Use derivatives" + defstr)
    parser.add_argument("-o", "--output_dir", dest="outdir",
                        help="Output directory base")
    parser.add_argument("-w", "--work_dir", dest="work_dir",
                        help="Output directory base")
    parser.add_argument("-p", "--plugin", dest="plugin",
                        default='Linear',
                        help="Plugin to use")
    parser.add_argument("--plugin_args", dest="plugin_args",
                        help="Plugin arguments")
    parser.add_argument("--sd", dest="subjects_dir",
                        help="FreeSurfer subjects directory (if available)")
    parser.add_argument('--surf_fwhm', default=15., dest='surf_fwhm',
                        type=float, help="Spatial FWHM" + defstr)
    parser.add_argument("--target_surfaces", dest="target_surfs", nargs="+",
                        default=['fsaverage4'],
                        help="FreeSurfer target surfaces" + defstr)
    parser.add_argument("--target", dest="target_file",
                        help=("Target in MNI space. Best to use the MindBoggle "
                              "template - only used with FreeSurfer"
                              "OASIS-30_Atropos_template_in_MNI152_2mm.nii.gz"))
    parser.add_argument("--sleep", dest="sleep", default=60., type=float,
                        help="Time to sleep between polls")
    args = parser.parse_args()
    outdir = args.outdir
    work_dir = os.getcwd()
    if args.work_dir:
        work_dir = os.path.abspath(args.work_dir)
    if outdir:
        outdir = os.path.abspath(outdir)
    else:
        outdir = os.path.join(work_dir, 'output')
    outdir = os.path.join(outdir, 'model%03d' % int(args.model),
                          'task%03d' % int(args.task))
    derivatives = args.derivatives
    if derivatives is None:
       derivatives = False
    wf = analyze_openfmri_dataset(data_dir=os.path.abspath(args.datasetdir),
                                  subject=args.subject,
                                  model_id=int(args.model),
                                  task_id=[int(args.task)],
                                  subj_prefix=args.subjectprefix,
                                  output_dir=outdir,
                                  hpcutoff=args.hpfilter,
                                  use_derivatives=derivatives,
                                  fwhm=args.fwhm,
                                  subjects_dir=args.subjects_dir,
                                  target=args.target_file,
                                  surf_fwhm=args.surf_fwhm,
                                  target_subject=args.target_surfs)
    #wf.config['execution']['remove_unnecessary_outputs'] = False

    wf.base_dir = work_dir
    wf.config['execution']['poll_sleep_duration'] = args.sleep
    #wf.config['exeuction']['stop_on_first_rerun'] = True
    wf.config['execution']['hash_method'] = 'timestamp'
    wf.write_graph(graph2use='flat')
    if args.plugin_args:
        wf.run(args.plugin, plugin_args=eval(args.plugin_args))
    else:
        wf.run(args.plugin)
    
