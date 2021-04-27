import math
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_pickle, unpack_data
import seaborn as sns
from mutual_information import (
    to_spikes_and_back,
    scoring_loop,
    plot_scores
)
sns.set()

step_factor_list = np.concatenate([np.linspace(0, 1, 10), np.linspace(1, 10, 5)])
target_dir = './plots/subjects/fmin0.5fmax2.5nperseg512/'

for subject_idx in range(9, 10):
    
    # track progress
    print('processing subject ' + str(subject_idx))

    # load ppg and hr data
    dataset = load_pickle(subject_idx)
    ppg, _, hr, activity, _= unpack_data(dataset)
    ppg = ppg[:-(64)]


    for evaluation_method in ['mutual_info', 'mutual_info_sklearn', 'regression_insample', 'regression_cv']:

        score_ppg, score_rec_list, rates_list = scoring_loop(
            [ppg], [hr],
            step_factor_list=step_factor_list,
            plot_detailed=False,
            evaluation_method=evaluation_method,
            n_neighbors=20,
            fmin=0.5,
            fmax=2.5,
            nperseg=512,
            noverlap=384
        )
        ylabel = 'Mean mutual information' if evaluation_method.startswith('mutual') else 'R2 score'
        plot_scores(score_ppg, score_rec_list, rates_list, ylabel, subject_idx, evaluation_method)
        if evaluation_method == 'regression_cv':
            plt.ylim(max(-0.5, min(score_rec_list) - 0.05), max(score_rec_list))
        plt.savefig(target_dir + 'subject_idx=' + str(subject_idx) + '-' + evaluation_method)
        plt.close()