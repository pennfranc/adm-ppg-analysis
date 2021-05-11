import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker

from data_loader import load_pickle, unpack_data
from mutual_information import (
    to_spikes_and_back,
    scoring_loop,
    plot_scores
)
sns.set()

step_factor_list = [0.1]
target_dir = './plots/all/fmin0.5fmax2.5ampnorm2/'
considered_subjects  = range(1, 16)
evaluation_methods = ['mutual_info_sklearn']

# load data
ppgs = []
hrs = []
activities_list = []
for subject_idx in considered_subjects:
    dataset = load_pickle(subject_idx)
    ppg, _, hr, activities, _= unpack_data(dataset)
    ppg = ppg[:-64]
    activities = activities[:-4]
    ppgs.append(ppg)
    hrs.append(hr)
    activities_list.append(activities)

activity_range = range(1, 9)


for activity_number in [-1]:#activity_range:
    
    # track progress
    print('processing activity ' + str(activity_number))


    for evaluation_method in evaluation_methods:
        print(evaluation_method)

        score_ppg, score_ppg_amp_normalized, score_rec_list, rates_list = scoring_loop(
            ppgs, hrs,
            activities_list=activities_list,
            chosen_activity=activity_number,
            step_factor_list=step_factor_list,
            plot_detailed=True,
            evaluation_method=evaluation_method,
            amp_norm_window_seconds=2,
            fmin=0.5,
            fmax=4,
            nperseg=512,
            noverlap=384
        )
