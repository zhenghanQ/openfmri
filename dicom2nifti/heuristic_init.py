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
    """

    #initialize a dictionary for all the scans you'll be grabbing
    info = {}


    return info
