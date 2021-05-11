import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_pickle, unpack_data
from mutual_information import scoring_loop_gaussian, plot_scores 

sns.set()

var_list = np.linspace(0.1, 10, 20)
target_dir = './plots/gaussian/'

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


for activity_number in [-1]:
    
    # track progress
    print('processing activity ' + str(activity_number))

    # load ppg and hr data
    dataset = load_pickle(subject_idx)
    ppg, _, hr, activity, _= unpack_data(dataset)
    ppg = ppg[:-(64)]

    for evaluation_method in ['mutual_info_sklearn', 'regression_cv']:

        score_ppg, score_rec_list = scoring_loop_gaussian(
            [ppg], [hr],
            activities_list=activities_list,
            chosen_activity=activity_number,
            var_list=var_list,
            plot_detailed=False,
            evaluation_method=evaluation_method,
            n_neighbors=20,
            fmin=0.5,
            fmax=2.5,
            nperseg=512,
            noverlap=384,
            order=1
        )
        ylabel = 'Mean mutual information' if evaluation_method.startswith('mutual') else 'R2 score'
        plt.plot(var_list, score_rec_list, label='rec signal')
        plt.axhline(score_ppg, label='original signal', color='orange')
        plt.legend()
        plt.ylabel(ylabel)
        plt.xlabel('Gaussian filter sigma')
        plt.title(evaluation_method + ' - Subject {}'.format(subject_idx))
        if evaluation_method == 'regression_cv':
            plt.ylim(max(-0.5, min(score_rec_list)), None)
        plt.savefig(target_dir + 'subject_idx=' + str(subject_idx) + '-' + evaluation_method)
        plt.close()