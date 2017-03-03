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

    rs = create_key('func/sub-{subject}_task-rest_run-{item:03d}_bold')
    dwi = create_key('dwi/sub-{subject}-run-{item:03d}_dwi')
    t1 = create_key('anat/sub-{subject}_run-{item:03d}_T1w')
    fm_rest = create_key('fmap/sub-{subject}_task-rest_run-{item:03d}_epi')
    rs_multi = create_key('func/sub-{subject}_task-mb_run-{item:03d}_bold')
    fm_task = create_key('fmap/sub-{subject}_task-bold_run-{item:03d}_epi')
    sent = create_key('func/sub-{subject}_task-sent_run{item:03d}_bold')
    srt=create_key('func/sub-{subject}_task-srt_run{item:03d}_bold')
    tone=create_key('func/sub-{subject}_task-tone_run{item:03d}_bold')
    nback=create_key('func/sub-{subject}_task-nback_run{item:03d}_bold')
    info = {rs:[], dwi:[], t1:[], fm_rest:[], rs_multi:[], fm_task:[], sent:[], srt:[], tone:[], nback:[]}
    
    for s in seqinfo:
        x,y,sl,nt = (s[6], s[7], s[8], s[9])
        if (nt == 149) and ('ep2d_Resting' in s[12]):
            info[rs] = [s[2]]
        elif (nt == 70) and ('DIFFUSION' in s[12]):
            info[dwi].append(s[2])
        elif (sl == 176) and (nt ==1) and ('T1_MPRAGE' in s[12]):
            info[t1]=[s[2]]
        elif ('field_mapping_3.5iso_32' in s[12]):
            info[fm_task].append(s[2])
        elif (nt==300) and ('Multiband' in s[12]):
            info[rs_multi].append(s[2])
        elif ('fm_rest' in s[12]):
            info[fm_rest].append(s[2])
        elif (nt == 191) and ('Sent' in s[12]):
            if not s[13]:
                info[sent].append(s[2])
        elif (nt == 232) and ('SRT' in s[12]):
            if not s[13]:
                info[srt].append(s[2])
        elif (nt >= 97) and ('cmrr' in s[12]):
            if not s[13]:
                info[tone].append(s[2])
        elif (nt == 267) and ('nBack' in s[12]):
            info[nback].append(s[2])
        else:
            pass
    return info
