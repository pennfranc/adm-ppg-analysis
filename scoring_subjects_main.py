import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from data_loader import load_pickle, unpack_data
import seaborn as sns
from scipy import stats
import pandas as pd
from mutual_information import (
    to_spikes_and_back,
    scoring_loop,
    plot_scores
)
sns.set()

subject_range = range(1, 16)
step_factor_list = np.concatenate([np.linspace(0, 1, 10), np.linspace(1, 10, 5)])
target_dir = './plots/subjects/fmin0.5fmax2.5nperseg512/'
compute_correlations = True
plot_bar_charts = True
plot_per_subject_bar_charts = False

mean_hrs = []
ppg_score_dict = {
    'mutual_info': [],
    'mutual_info_sklearn': [],
    'regression_insample': [],
    'regression_cv': []
}

ppg_amp_norm_dict = {
    'mutual_info': [],
    'mutual_info_sklearn': [],
    'regression_insample': [],
    'regression_cv': []
}

frts_dict = {
    'mutual_info': [],
    'mutual_info_sklearn': [],
    'regression_insample': [],
    'regression_cv': []
}

max_score_dict = {
    'mutual_info': [],
    'mutual_info_sklearn': [],
    'regression_insample': [],
    'regression_cv': []
}

for subject_idx in subject_range:
    
    # track progress
    print('processing subject ' + str(subject_idx))

    # load ppg and hr data
    dataset = load_pickle(subject_idx)
    ppg, _, hr, activity, _= unpack_data(dataset)
    ppg = ppg[:-(64)]

    mean_hrs.append(np.mean(hr))


    for evaluation_method in ['mutual_info', 'mutual_info_sklearn', 'regression_insample', 'regression_cv']:

        score_ppg, score_ppg_amp_normalized, score_rec_list, rates_list = scoring_loop(
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

        # save ppg score
        ppg_score_dict[evaluation_method].append(score_ppg)

        # save ppg amp norm score
        ppg_amp_norm_dict[evaluation_method].append(score_ppg_amp_normalized)

        # save FRT
        if sum((score_rec_list >= score_ppg).astype(int)) > 0:
            frt = rates_list[np.where(score_rec_list >= score_ppg)[0].max()]
            frts_dict[evaluation_method].append(frt)
        else:
            frts_dict[evaluation_method].append(np.nan)

        # save maximum rec score
        max_score_dict[evaluation_method].append(max(score_rec_list))

        ylabel = 'Mean mutual information' if evaluation_method.startswith('mutual') else 'R2 score'
        plot_scores(score_ppg, score_ppg_amp_normalized, score_rec_list, rates_list, ylabel, subject_idx, evaluation_method)
        if evaluation_method == 'regression_cv':
            plt.ylim(max(-0.5, min(score_rec_list) - 0.05), max(score_rec_list))
        plt.savefig(target_dir + 'subject_idx=' + str(subject_idx) + '-' + evaluation_method)
        plt.close()

if compute_correlations:
    for evaluation_method in ['mutual_info', 'mutual_info_sklearn', 'regression_insample', 'regression_cv']:
        
        plt.scatter(mean_hrs, frts_dict[evaluation_method])
        plt.xlabel('Mean heart rate [Hz]')
        plt.ylabel('Minimum required mean firing rate [Hz]')
        plt.title(evaluation_method + ' - faithful reconstruction threshold')
        plt.savefig(target_dir + 'frt' + '-' + evaluation_method)
        plt.close()

        non_nan_mask = ~np.isnan(np.array(frts_dict[evaluation_method]))
        correlation = stats.spearmanr(np.array(mean_hrs)[non_nan_mask], np.array(frts_dict[evaluation_method])[non_nan_mask])
        print('Spearman correlation of {}: {}'.format(evaluation_method, correlation))


if plot_bar_charts:
    for evaluation_method in ['mutual_info', 'mutual_info_sklearn', 'regression_insample', 'regression_cv']:
        


        # plot absolute max ADM performance
        width = 0.3
        plt.bar(np.array(subject_range) - width / 2, np.array(max_score_dict[evaluation_method]) - np.array(ppg_score_dict[evaluation_method]), label='ADM reconstruction', width=width)
        plt.bar(np.array(subject_range) + width / 2, np.array(ppg_amp_norm_dict[evaluation_method]) - np.array(ppg_score_dict[evaluation_method]), label='PPG amp normalization', width=width)
        plt.xlabel('Subject ID')
        plt.ylabel('Absolute similarity score improvement')
        plt.title(evaluation_method + ' - ADM and amp normalization - score improvements')
        plt.legend()
        plt.gca().xaxis.set_major_locator(mticker.FixedLocator(subject_range))
        plt.savefig(target_dir + 'absolute-improvement' + '-' + evaluation_method)
        plt.close()

        # plot ppg performance
        plt.bar(subject_range, np.array(ppg_score_dict[evaluation_method]))
        plt.xlabel('Subject ID')
        plt.ylabel('PPG Score')
        plt.title(evaluation_method + ' - PPG score')
        plt.gca().xaxis.set_major_locator(mticker.FixedLocator(subject_range))
        plt.savefig(target_dir + 'ppg-score' + '-' + evaluation_method)
        plt.close()

if plot_per_subject_bar_charts:
    
    for subject_idx in subject_range:

        plt.bar(
            ['Raw PPG MI', 'Best ADM MI', 'Raw PPG R2', 'Best ADM R2'],
            [
                ppg_score_dict['mutual_info_sklearn'][subject_idx-1],
                max_score_dict['mutual_info_sklearn'][subject_idx-1],
                ppg_score_dict['regression_cv'][subject_idx-1],
                max_score_dict['regression_cv'][subject_idx-1]
            ]
        )

        plt.title('Subject {}'.format(subject_idx))
        plt.ylabel('Score')
        plt.savefig(target_dir + 'per_subject_bar_charts/subject-' + str(subject_idx))
        plt.close()