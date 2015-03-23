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

    rs = create_key('resting/rest/bold', outtype=('dicom', 'nii.gz'))
    spin = create_key('resting/rest/se', outtype=('dicom', 'nii.gz'))
    #dwi = create_key('dmri/dwi_{item:03d}', outtype=('dicom',))
    dwi7ap = create_key('dmri/dwi_1k_AP', outtype=('dicom', 'nii.gz'))
    dwi72pa = create_key('dmri/dwi_1k_PA', outtype=('dicom', 'nii.gz'))
    t1 = create_key('anatomy/T1_{item:03d}')
    t2 = create_key('anatomy/T2_{item:03d}')
    pataka = create_key('BOLD/task001_run{item:03d}/bold')
    sentences=create_key('BOLD/task002_run{item:03d}/bold')
    nonwordrep=create_key('BOLD/task003_run{item:03d}/bold')
    facematch=create_key('BOLD/task004_run{item:03d}/bold')
    emosent=create_key('BOLD/task005_run{item:03d}/bold')
    vowels=create_key('BOLD/task006_run{item:03d}/bold')
    pitch_emph=create_key('BOLD/task007_run{item:03d}/bold')
    movie_trailer=create_key('BOLD/task008_run{item:03d}/bold')

    info = {rs: [], spin: [], dwi7ap:[], dwi72pa:[], t1:[], t2:[], pataka:[],
            sentences:[], nonwordrep:[], facematch:[], emosent:[], vowels:[],
            pitch_emph:[], movie_trailer:[]}
    last_run = len(seqinfo)
    for s in seqinfo:
        x,y,sl,nt = (s[6], s[7], s[8], s[9])
        if (nt == 300) and ('SMS5_rsfMRI' in s[12]):
            info[rs] = [s[2]]
        elif (nt == 4) and ('Spin_Echo' in s[12]):
            info[spin].append(s[2])
        #elif (sl > 1) and (nt == 72) and ('SMS2-diff_b1000' in s[12]):
        #    info[dwi].append(s[2])
        #elif (sl > 1) and (nt == 7 ) and ('SMS2-diff_b100_free' in s[12]):
        #    info[dwi].append(s[2])
        elif (sl == 176) and (nt ==1) and ('T1_MPRAGE' in s[12]):
            info[t1].append(s[2])
        elif (nt > 175) and ('PaTaKa' in s[12]):
            info[pataka].append(s[2])
        elif (nt == 64) and ('Sentences' in s[12]):
            info[sentences].append(s[2])
        elif (nt == 42 ) and ('Nonword' in s[12]):
            info[nonwordrep].append(s[2])
        elif (nt == 99) and ('FaceMatch' in s[12]):
            info[facematch].append(s[2])
        elif (nt == 48) and ('EmoSent' in s[12]):
            info[emosent].append(s[2])
        elif (nt == 60) and ('Vowels' in s[12]):
            info[vowels].append(s[2])
        elif (nt == 101) and ('PitchEmph' in s[12]):
            info[pitch_emph].append(s[2])
        elif (nt == 138) and ('Movie' in s[12]):
            info[movie_trailer].append(s[2])
        elif (sl == 176) and (nt == 1 ) and ('T2_SPACE' in s[12]):
            info[t2] = [s[2]]
	elif (nt == 72) and ('SMS2' in s[12]):
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
