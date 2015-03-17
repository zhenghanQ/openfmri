"""
Initial heuristic. Returns info.
"""

import os

def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return (template, outtype, annotation_classes)

def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where

    allowed template fields - follow python string module: 

    item: index within category 
    subject: participant id 
    seqitem: run number during scanning
    subindex: sub index within group
    """
    #  Create the T1 anatomy file
    t1 = create_key('anatomy/T1_{item:03d}')

    #  Pull out the non word reptition fMRI run
    nwr = create_key('BOLD/task001_run{item:03d}/bold')

    #initialize a dictionary for all the scans you'll be grabbing
    info = {t1:[],nwr:[]} 

    # commented out because not needed anymore
    #info={}

    # the basic "if" statement structure is a hack job customized to every project
    # the first part of the if condition filters by some "unique-ish" number from columns 6,7,8, and 9
    # the second, and, part of the condition filters by hopefully a meaningful sequence name that was used in the scanner.

    #  Note:  the t1 image is NOT appended to info.  Keep only the very last t1 run in the file.
    #  Note:  the functional sequences use .append when adding values to the dictionary key.  This is because multiple runs will have the same MRI sequence name, but we want all of them.
    #  Note:  The 13th column is a boolean that refers to motion correction.  Only keep runs where the boolean is false.

    last_run = len(seqinfo)
    for s in seqinfo:
        x,y,sl,nt = (s[6], s[7], s[8], s[9])
        if (sl == 176) and (nt ==1) and ('T1_MPRAGE' in s[12]):
            info[t1]=[s[2]]
        elif (nt == 42) and ('Nonword-Repetition' in s[12]):
        if not s[13]:
                info[nwr].append(s[2])
    else:
            pass
    return info
