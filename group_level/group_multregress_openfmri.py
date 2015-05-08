"""
group openfmri script
reflective of openfmri organization in draft document Open_Brain_Imaging_Data_Structure.pdf date 2015-03-18
exception: use of txt files instead of tsvs for the time being
"""

import os
from nipype import config
config.enable_provenance()
from nipype import Workflow, Node, MapNode
from nipype import DataGrabber, DataSink
from nipype.interfaces.fsl import (Merge, FLAMEO, ContrastMgr,
                                   SmoothEstimate, Cluster, ImageMaths, MultipleRegressDesign)
import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as util
from nipype.interfaces.fsl.maths import BinaryMaths
get_len = lambda x: len(x)

def l1_contrasts_num(model_id,
                  task_id,
                  dataset_dir):
    import numpy as np
    import os
    contrast_def = []
    contrasts=0
    contrast_file = os.path.join(dataset_dir, 'models', 'model%03d' % model_id,
                                 'task_contrasts.txt')
    if os.path.exists(contrast_file):
        with open(contrast_file, 'rt') as fp:
            contrast_def.extend([np.array(row.split()) for row in fp.readlines() if row.strip()])
    for row in contrast_def:
        if row[0] != 'task%03d' % task_id:
            continue
        contrasts=contrasts+1
    condition_info = []
    cond_file = os.path.join(dataset_dir, 'models', 'model%03d' % model_id,
                             'condition_key.txt')
    with open(cond_file, 'rt') as fp:
        for line in fp:
            info = line.strip().split()
            condition_info.append([info[0], info[1], ' '.join(info[2:])])
    for row in condition_info:
        if row[0] != 'task%03d' % task_id:
            continue
        contrasts=contrasts+1
        cope_id=range(1,contrasts+1)
    return cope_id

def get_sub_vars(dataset_dir,task_id,model_id):
    import numpy as np
    import os
    subs_list_def=[]
    sub_list_file=os.path.join(dataset_dir,'groups','participant_key.txt')
    with open(sub_list_file, 'rt') as fp:
        subs_list_def.extend([np.array(row.split()) for row in fp.readlines() if row.strip()])
        task_id_num=int(task_id)
        subs_list=[y[0] for y in subs_list_def[1:]]
        groups=[1 for y in subs_list_def[1:]]
        #groups=[int(y[task_id_num]) for y in subs_list_def[1:]]
        #subs_list=[str(x) for x in [y[0] for y in subs_list_def[1:]] #if subs_list_def[task_id_num][x] != 0]
        #groups=[int(x) for x in subs_list_def[task_id_num][:] if  x != 0]
        behav_file = os.path.join(dataset_dir,'groups','behav.txt')
        regressors={}
        behav_list_def=[]
        with open(behav_file,'rt') as fp:
             behav_list_def.extend([np.array(row.split()) for row in fp.readlines() if row.strip()])
             behav_ids=[x[0] for x in behav_list_def[1:]]
             for regressor_name in behav_list_def[0][1:]:
                 regressors[regressor_name]=[]
             for sub in subs_list:
                 if sub in behav_ids:
                     for key in regressors.keys():
                         idx=np.where(behav_list_def[0][:]==key)[0][0]
                         for row in behav_list_def:
                             if row[0] == sub:
                                 regressors[key].append(float(row[idx]))
                 else:
                     raise Exception('%s is missing from behav.txt'%sub)
    contrast_def=[]
    group_contrast_file=os.path.join(dataset_dir,'groups','contrasts.txt')
    with open(group_contrast_file, 'rt') as fp:
            contrast_def.extend([np.array(row.split()) for row in fp.readlines() if row.strip()])
    
    
    contrasts = []
    for row in contrast_def:
        if row[0] != 'task%03d' % task_id:
            continue
        con = [tuple([row[1], 'T', eval(row[2]),
            row[3:].astype(float).tolist()])]
        contrasts.append(con)
    
    regressors_needed=[]
    for idx,con in enumerate(contrasts):
        model_regressor={}
        for cond in con[0][2]:
            model_regressor[cond]=regressors[cond]
        regressors_needed.append(model_regressor)   
    return regressors_needed,contrasts,groups

def subj_list_gen(dataset_dir):#, model_id, task_id):
    import numpy as np
    import os
    behav_file = os.path.join(dataset_dir,'groups','behav.txt')
    behav_list_def=[]
    with open(behav_file,'rt') as fp:
         behav_list_def.extend([np.array(row.split()) for row in fp.readlines() if row.strip()])
         subj_list=[x[0] for x in behav_list_def[1:]]
    return subj_list

def group_multregress_openfmri(dataset_dir,model_id=None,task_id=None,l1output_dir=None,out_dir=None, no_reversal=False, plugin=None, plugin_args=None):
    
    num_copes=l1_contrasts_num(model_id,task_id,dataset_dir)
    
    regressors_needed,contrasts,groups=get_sub_vars(dataset_dir,task_id,model_id)
    
    subj_list=subj_list_gen(dataset_dir)    
    
    for idx,contrast in enumerate(contrasts):
        wk = Workflow(name='mult_regress')
        wk.base_dir = os.path.join(work_dir,'group','model_%03d_task_%03d_contrast_%s'%(model_id,task_id,contrast[0][0]))
        #wk=Workflow(name='model_%03d_task_%03d_contrast_%s'%(model_id,task_id,contrast[1])

        info = Node(util.IdentityInterface(fields=['model_id','task_id','dataset_dir','subj_list']),
                                            name='infosource')
        info.inputs.model_id=model_id
        info.inputs.task_id=task_id
        info.inputs.dataset_dir=dataset_dir

        dg = Node(DataGrabber(infields=['model_id','task_id','cope_id'],
                              outfields=['copes', 'varcopes']),name='grabber')
        dg.inputs.template = os.path.join(l1output_dir,'model%03d/task%03d/%s/%scopes/mni/%scope%02d.nii.gz')
        #dg.inputs.template = os.path.join(l1output_dir,'model%03d/task%03d/%s/%scopes/mni/model%03d/task%03d_%scope%02d.nii.gz')
        dg.inputs.template_args['copes'] = [['model_id','task_id',subj_list,'','', 'cope_id']]
        dg.inputs.template_args['varcopes'] = [['model_id','task_id',subj_list,'var', 'var', 'cope_id']]
       # dg.inputs.template_args['copes'] = [['model_id','task_id',subj_list,'','model_id','task_id','','cope_id']]
       # dg.inputs.template_args['varcopes'] = [['model_id','task_id',subj_list,'var','model_id','task_id','var','cope_id']]    
        dg.iterables=('cope_id',num_copes)

        dg.inputs.sort_filelist = False

        wk.connect(info,'model_id',dg,'model_id')
        wk.connect(info,'task_id',dg,'task_id')

        regressors_needed,contrasts,groups=get_sub_vars(dataset_dir,task_id,model_id)
    
        model=Node(MultipleRegressDesign(),name='l2model')
        #model.iterables=[('regressors',regressors_needed),('contrasts',contrasts)]
        model.inputs.groups=groups
        model.inputs.contrasts=contrasts[idx]
        model.inputs.regressors=regressors_needed[idx]

        mergecopes = Node(Merge(dimension='t'), name='merge_copes')
        wk.connect(dg, 'copes', mergecopes, 'in_files')

        mergevarcopes = Node(Merge(dimension='t'), name='merge_varcopes')
        wk.connect(dg, 'varcopes', mergevarcopes, 'in_files')

        mask_file = fsl.Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz')
        flame = Node(FLAMEO(), name='flameo')
        flame.inputs.mask_file =  mask_file
        flame.inputs.run_mode = 'flame1'

        wk.connect(model, 'design_mat', flame, 'design_file')
        wk.connect(model, 'design_con', flame, 't_con_file')
        wk.connect(mergecopes, 'merged_file', flame, 'cope_file')
        wk.connect(mergevarcopes, 'merged_file', flame, 'var_cope_file')
        wk.connect(model, 'design_grp', flame, 'cov_split_file')

        smoothest = Node(SmoothEstimate(), name='smooth_estimate')
        wk.connect(flame, 'zstats', smoothest, 'zstat_file')
        smoothest.inputs.mask_file = mask_file

  
        cluster = Node(Cluster(), name='cluster')
        wk.connect(smoothest,'dlh', cluster, 'dlh')
        wk.connect(smoothest, 'volume', cluster, 'volume')
        cluster.inputs.connectivity = 26
        cluster.inputs.threshold=2.3
        cluster.inputs.pthreshold = 0.01
        cluster.inputs.out_threshold_file = True
        cluster.inputs.out_index_file = True
        cluster.inputs.out_localmax_txt_file = True

        wk.connect(flame, 'zstats', cluster, 'in_file')
    
        ztopval = Node(ImageMaths(op_string='-ztop', suffix='_pval'),
                       name='z2pval')
        wk.connect(flame, 'zstats', ztopval,'in_file')
    
    

        sinker = Node(DataSink(), name='sinker')
        sinker.inputs.base_directory = os.path.join(out_dir,'contrast_%s'%contrast[0][0])
        sinker.inputs.substitutions = [('_cope_id', 'contrast'),
                                    ('_maths_', '_reversed_')]
        if no_reversal == False:
            zstats_reverse = Node( BinaryMaths()  , name='zstats_reverse')
            zstats_reverse.inputs.operation = 'mul'
            zstats_reverse.inputs.operand_value= -1
            wk.connect(flame, 'zstats', zstats_reverse, 'in_file')

            cluster2=cluster.clone(name='cluster2')
            wk.connect(smoothest,'dlh',cluster2,'dlh')
            wk.connect(smoothest,'volume',cluster2,'volume')
            wk.connect(zstats_reverse,'out_file',cluster2,'in_file')
   
            ztopval2 = ztopval.clone(name='ztopval2')
            wk.connect(zstats_reverse,'out_file',ztopval2,'in_file')
            wk.connect(flame, 'zstats', sinker, 'stats')
            wk.connect(cluster, 'threshold_file', sinker, 'stats.@thr')
            wk.connect(cluster, 'index_file', sinker, 'stats.@index')
            wk.connect(cluster, 'localmax_txt_file', sinker, 'stats.@localmax')

            wk.connect(zstats_reverse,'out_file',sinker,'stats.@neg')
            wk.connect(cluster2,'threshold_file',sinker,'stats.@neg_thr')
            wk.connect(cluster2,'index_file',sinker,'stats.@neg_index')
            wk.connect(cluster2,'localmax_txt_file',sinker,'stats.@neg_localmax')
        if plugin_args:
            wk.run(plugin, plugin_args=plugin_args)
        else:
            wk.run(plugin)   
    return 

if __name__ == '__main__':
    import argparse
    defstr = ' (default %(default)s)'
    parser = argparse.ArgumentParser(prog='group_multregress_openfmri.py',
                                     description=__doc__)
    parser.add_argument('-m', '--model', default=1,
                        help="Model index" + defstr)
    parser.add_argument('-t', '--task', default=1,
                        type=int, help="Task index" + defstr)
    parser.add_argument("-o", "--output_dir", dest="outdir",
                        help="Output directory base")
    parser.add_argument('-d', '--datasetdir', required=True)
    parser.add_argument("-l1", "--l1_output_dir", dest="l1out_dir",
                        help="l1_output directory ")
    parser.add_argument("-w", "--work_dir", dest="work_dir",
                        help="Output directory base")
    parser.add_argument("-p", "--plugin", dest="plugin",
                        default='Linear',
                        help="Plugin to use")
    parser.add_argument("--plugin_args", dest="plugin_args",
                        help="Plugin arguments")
    parser.add_argument("--norev",action='store_true',
                        help="if reversal of contrasts already in task_contrasts.txt")
    args = parser.parse_args()
    outdir = args.outdir
    work_dir = os.getcwd()

    if args.work_dir:
        work_dir = os.path.abspath(args.work_dir)
    if args.outdir:
        outdir = os.path.abspath(outdir)
    if args.l1out_dir:
        l1_outdir=os.path.abspath(args.l1out_dir)
    else:
        l1_outdir=os.path.join(args.datasetdir,'l1output')

    outdir = os.path.join(outdir, 'group','model%03d' % int(args.model),
                          'task%03d' % int(args.task))

    wf = group_multregress_openfmri(model_id=int(args.model),
                                  task_id=int(args.task),
                                  l1output_dir=l1_outdir,
                                  out_dir=outdir,
                                  dataset_dir=os.path.abspath(args.datasetdir),
                                  no_reversal=args.norev,
                                  plugin=args.plugin,
                                  plugin_args=eval(args.plugin_args))
