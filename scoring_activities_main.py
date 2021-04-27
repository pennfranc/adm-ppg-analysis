import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_pickle, unpack_data
from mutual_information import (
    to_spikes_and_back,
    score_pipeline,
    scoring_loop,
    plot_scores
)
sns.set()

step_factor_list = np.concatenate([np.linspace(0, 1, 10), np.linspace(1, 10, 5)])
target_dir = './plots/all/fmin0.5fmax2.5nperseg512/'
considered_subjects  = range(1, 16)

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


for activity_number in [-1]: #range(1, 9):
    
    # track progress
    print('processing activity ' + str(activity_number))


    for evaluation_method in ['mutual_info', 'mutual_info_sklearn', 'regression_insample', 'regression_cv']:

        score_ppg, score_rec_list, rates_list = scoring_loop(
            ppgs, hrs,
            activities_list=activities_list,
            chosen_activity=activity_number,
            step_factor_list=step_factor_list,
            plot_detailed=False,
            evaluation_method=evaluation_method,
            fmin=0.5,
            fmax=2.5,
            nperseg=512,
            noverlap=384
        )
        ylabel = 'Mean mutual information' if evaluation_method.startswith('mutual') else 'R2 score'
        plot_scores(score_ppg, score_rec_list, rates_list, ylabel, activity_number, evaluation_method)
        if evaluation_method == 'regression_cv':
            plt.ylim(max(-0.5, min(score_rec_list) - 0.05), max(score_rec_list))
        plt.savefig(target_dir + 'activity_number=' + str(activity_number) + '-' + evaluation_method)
        plt.close()