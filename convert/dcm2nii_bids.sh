#!/bin/bash
SUBJECT=$1
heudiconv -d /mindhive/xnat/dicom_storage/CASL/%s/dicom/*.dcm -o /mindhive/xnat/data/CASL/bids -f heuristic_CASL_bids.py -c dcm2niix -q om_interactive -s $SUBJECT -b

