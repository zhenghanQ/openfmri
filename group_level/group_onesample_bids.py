import os
from nipype import config
config.enable_provenance()
from nipype import Workflow, Node, MapNode
from nipype import DataGrabber, DataSink
from nipype.interfaces.fsl import (L2Model, Merge, FLAMEO, ContrastMgr, 
                                   SmoothEstimate, Cluster, ImageMaths)
import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as util
from nipype.interfaces.fsl.maths import BinaryMaths
get_len = lambda x: len(x)
def contrasts_num(model_id,
                  task_id,
                  dataset_dir):
    import numpy as np
    import os
    contrast_def = []
    contrasts = 0
    task_key = os.path.join(dataset_dir, 'tasks.tsv')
    tasks = []
    if os.path.exists(task_key):
        with open(task_key, 'rt') as fp:
            tasks.extend([np.array(row.split('\t')[1].rsplit()[0]) for row in fp.readlines()])
    contrast_file = os.path.join(dataset_dir, 'code', 'model', 'model%03d' % model_id,
                                 'task_contrasts.txt')
    if os.path.exists(contrast_file):
        with open(contrast_file, 'rt') as fp:
            contrast_def.extend([np.array(row.split()) for row in fp.readlines() if row.strip()])
    for row in contrast_def:
        if row[0] != "task-%s" % tasks[task_id-1]:
            continue
        contrasts = contrasts + 1
    cope_id = range(1, contrasts + 1)
    return cope_id

def group_onesample_openfmri(dataset_dir,model_id=None,task_id=None,l1output_dir=None,out_dir=None, no_reversal=False):

    wk = Workflow(name='one_sample')
    wk.base_dir = os.path.abspath(work_dir)

    info = Node(util.IdentityInterface(fields=['model_id','task_id','dataset_dir']),
                                        name='infosource')
    info.inputs.model_id=model_id
    info.inputs.task_id=task_id
    info.inputs.dataset_dir=dataset_dir
    
    num_copes=contrasts_num(model_id,task_id,dataset_dir)

    dg = Node(DataGrabber(infields=['model_id','task_id','cope_id'], 
                          outfields=['copes', 'varcopes']),name='grabber')
    dg.inputs.template = os.path.join(l1output_dir,'model%03d/task%03d/*/%scopes/mni/%scope%02d.nii.gz')
    dg.inputs.template_args['copes'] = [['model_id','task_id','', '', 'cope_id']]
    dg.inputs.template_args['varcopes'] = [['model_id','task_id','var', 'var', 'cope_id']]
    dg.iterables=('cope_id',num_copes)

    dg.inputs.sort_filelist = True

    wk.connect(info,'model_id',dg,'model_id')
    wk.connect(info,'task_id',dg,'task_id')

    model = Node(L2Model(), name='l2model')

    wk.connect(dg, ('copes', get_len), model, 'num_copes')

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
    cluster.inputs.pthreshold = 0.05
    cluster.inputs.out_threshold_file = True
    cluster.inputs.out_index_file = True
    cluster.inputs.out_localmax_txt_file = True

    wk.connect(flame, 'zstats', cluster, 'in_file')
	 
    ztopval = Node(ImageMaths(op_string='-ztop', suffix='_pval'),
                   name='z2pval')
    wk.connect(flame, 'zstats', ztopval,'in_file')
    
    

    sinker = Node(DataSink(), name='sinker')  
    sinker.inputs.base_directory = os.path.abspath(out_dir)
    sinker.inputs.substitutions = [('_cope_id', 'contrast'),
			            ('_maths__', '_reversed_')]
    
    wk.connect(flame, 'zstats', sinker, 'stats')
    wk.connect(cluster, 'threshold_file', sinker, 'stats.@thr')
    wk.connect(cluster, 'index_file', sinker, 'stats.@index')
    wk.connect(cluster, 'localmax_txt_file', sinker, 'stats.@localmax')
    
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

        wk.connect(zstats_reverse,'out_file',sinker,'stats.@neg')
        wk.connect(cluster2,'threshold_file',sinker,'stats.@neg_thr')
        wk.connect(cluster2,'index_file',sinker,'stats.@neg_index')
        wk.connect(cluster2,'localmax_txt_file',sinker,'stats.@neg_localmax')

    return wk

if __name__ == '__main__':
    import argparse
    defstr = ' (default %(default)s)'
    parser = argparse.ArgumentParser(prog='group_onesample_openfmri.py',
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

    outdir = os.path.join(outdir, 'one_sample','model%03d' % int(args.model),
                          'task%03d' % int(args.task))

    wf = group_onesample_openfmri(model_id=int(args.model),
                                  task_id=int(args.task),
                                  l1output_dir=l1_outdir,
                                  out_dir=outdir,
                                  dataset_dir=os.path.abspath(args.datasetdir),
                                  no_reversal=args.norev)
    wf.base_dir = work_dir
    if args.plugin_args:
        wf.run(args.plugin, plugin_args=eval(args.plugin_args))
    else:
        wf.run(args.plugin)
