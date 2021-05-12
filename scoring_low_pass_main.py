import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_pickle, unpack_data
from mutual_information import scoring_loop_lowpass, plot_scores 

sns.set()

cutoff_freq_list = [1.0, 1.2, 1.4, 2.0, 2.5, 3.0]
target_dir = './plots/lowpass/'

plot_spike_dependence = True
considered_subjects = range(1, 16)

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

plt.figure(figsize=(16,10),dpi=150)
for activity_number in [-1]:
    
    # track progress
    print('processing activity ' + str(activity_number))

    for evaluation_method in ['mutual_info_sklearn', 'regression_cv']:

        score_ppg, score_rec_list = scoring_loop_lowpass(
            ppgs, hrs,
            activities_list=activities_list,
            chosen_activity=activity_number,
            cutoff_freq_list=cutoff_freq_list,
            plot_detailed=True,
            evaluation_method=evaluation_method,
            fmin=0.5,
            fmax=4.0,
            nperseg=512,
            noverlap=384,
            order=1
        )

        if plot_spike_dependence:
            ylabel = 'Mean mutual information' if evaluation_method.startswith('mutual') else 'R2 score'
            plt.plot(cutoff_freq_list, score_rec_list, label='rec signal')
            plt.axhline(score_ppg, label='original signal', color='orange')
            plt.legend()
            plt.ylabel(ylabel)
            plt.xlabel('Cutoff frequency [Hz]')
            plt.title(evaluation_method)
            if evaluation_method == 'regression_cv':
                plt.ylim(max(-0.5, min(score_rec_list)), None)
            plt.savefig(target_dir + evaluation_method)
            plt.close()