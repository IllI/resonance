import os
import sys
sys.path.append('c:/Users/cityz/IllI/newer_all/python')
try:
    from align_multi_subject_geometry import process_subject
except ImportError:
    pass

for s in ['sub-1', 'sub-2', 'sub-3', 'sub-4', 'sub-5', 'sub-6']:
    path = os.path.join('c:/Users/cityz/IllI/newer_all/data/openneuro/ds000105', s)
    te_ts_2d, te_labels, _, te_TR = process_subject(path)
    print(s, set(te_labels))
