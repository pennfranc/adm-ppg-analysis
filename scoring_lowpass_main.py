import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_pickle, unpack_data
from mutual_information import (
    to_spikes_and_back,
    first_order_low_pass,
    score_pipeline,
    scoring_loop,
    scoring_loop_low_pass,
    plot_scores
)
sns.set()

step_factor_list = np.concatenate([np.linspace(0, 1, 10), np.linspace(1, 10, 5)])
cutoff_freq_list = np.linspace(0.01, 10, 20)
target_dir = './plots/lowpass/'

for subject_idx in range(3, 4):
    
    # track progress
    print('processing subject ' + str(subject_idx))

    # load ppg and hr data
    dataset = load_pickle(subject_idx)
    ppg, _, hr, activity, _= unpack_data(dataset)
    ppg = ppg[:-(64)]


    for evaluation_method in ['mutual_info', 'mutual_info_sklearn', 'regression_insample', 'regression_cv']:

        score_ppg, score_rec_list = scoring_loop_low_pass(
            ppg, hr,
            cutoff_freq_list=cutoff_freq_list,
            plot_detailed=False,
            evaluation_method=evaluation_method,
            n_neighbors=20
        )
        ylabel = 'Mean mutual information' if evaluation_method.startswith('mutual') else 'R2 score'
        plt.plot(cutoff_freq_list, score_rec_list, label='rec signal')
        plt.axhline(score_ppg, label='original signal', color='orange')
        plt.legend()
        plt.ylabel(ylabel)
        plt.xlabel('Cutoff frequency [Hz]')
        plt.title(evaluation_method + ' - Subject {}'.format(subject_idx))
        if evaluation_method == 'regression_cv':
            plt.ylim(-0.5, 1)
        plt.savefig(target_dir + 'subject_idx=' + str(subject_idx) + '-' + evaluation_method)
        plt.close()