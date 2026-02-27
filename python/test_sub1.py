import os
import sys
sys.path.append('c:/Users/cityz/IllI/newer_all/python')
try:
    from align_multi_subject_geometry import process_subject
except ImportError:
    pass

path = os.path.join('c:/Users/cityz/IllI/newer_all/data/openneuro/ds000105', 'sub-1')
te_ts_2d, te_labels, _, te_TR = process_subject(path)
print(set(te_labels))
