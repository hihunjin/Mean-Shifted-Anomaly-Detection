import os
import pandas as pd
import numpy as np

settings = open('celeA_two_class_settings.txt', 'r')
lines_settings = settings.readlines()

rows = []
for line in lines_settings:
    _line = line.split()
    rows.append(
        {
            'dataset': _line[0].split("=")[-1],
            "backbone": _line[1].split("=")[-1],
            "target_index": _line[2].split("=")[-1],
            "is_angular": _line[3].split("=")[-1],
        }
    )

epochs = open('celeA_two_class_auroc.txt', 'r')
lines = epochs.readlines()

num_row = 0
for line in lines:
    s_line = line.split()
    epoch_num, auroc = int(s_line[1][:-1]), float(s_line[-1])
    if epoch_num == 0:
        aurocs = [auroc]
    elif epoch_num == 20:
        aurocs = [auroc]
        epoch_best, epoch_last = max(aurocs), auroc
        rows[num_row]['epoch_best'] = epoch_best
        rows[num_row]['epoch_last'] = epoch_last
        num_row += 1
    else:
        aurocs.append(auroc)

df = pd.DataFrame(rows)
df.to_csv('results_celeA_two_class.csv', index=False)
