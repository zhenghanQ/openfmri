This repository aggregates Gablab scripts that operate in the OpenfMRI framework.

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
-l1 /path/to/l1output/dir/ -w /path/to/working/dir/ -p 'SLURM' --plugin_args [plugin arguments] [--norev]
```

Or type in `python group_multregress_openfmri.py` at the command line to see usage. Specify the --norev option if reversal of contrasts is already in your contrasts.txt file; otherwise, the script will include outputs for reversal of contrasts by default.

Example:
```
python group_multregress_openfmri.py -m 1 -t 1 -o /om/project/simba/analysis/openfmri/l2output/ 
-d /om/project/simba/analysis/openfmri/ -l1 /om/project/simba/analysis/openfmri/l1output/
-w /om/scratch/Sun/annepark/SIMBA/fmri/ -p 'SLURM' --plugin_args "{'sbatch_args':'-p om_all_nodes'}"
```
