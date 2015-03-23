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

    #dwi = create_key('dmri/dwi_{item:03d}', outtype=('dicom',))
    dwi7ap = create_key('dmri/dwi_1k_AP', outtype=('dicom', 'nii.gz'))
    dwi72pa = create_key('dmri/dwi_1k_PA', outtype=('dicom', 'nii.gz'))


    info = {dwi7ap:[], dwi72pa:[]}
    last_run = len(seqinfo)
    for s in seqinfo:
        x,y,sl,nt = (s[6], s[7], s[8], s[9])
    	if (nt == 72) and ('SMS2' in s[12]):
                if not s[13]:
                    if 'PA' in s[12]:
                        info[dwi72pa].append(s[2])
                    else:
                        info[dwi72ap].append(s[2])
    	elif (nt == 7) and ('SMS2' in s[12]):
                if not s[13]:
                    if 'AP' in s[12]:
                        info[dwi7ap].append(s[2])
	else:
            pass
    return info
