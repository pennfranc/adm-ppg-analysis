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

step_factor_list = np.concatenate([np.linspace(0, 1, 10), np.linspace(1, 10, 5)])
target_dir = './plots/all/fmin0.5fmax2.5nperseg512/'
considered_subjects  = range(1, 16)
plot_bar_charts = True
plot_per_activity_bar_charts = False

mean_hrs = []
ppg_score_dict = {
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


for activity_number in [-1]: # activity_range:
    
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

        # save ppg score
        ppg_score_dict[evaluation_method].append(score_ppg)

        # save FRT
        if sum((score_rec_list >= score_ppg).astype(int)) > 0:
            frt = rates_list[np.where(score_rec_list >= score_ppg)[0].max()]
            frts_dict[evaluation_method].append(frt)
        else:
            frts_dict[evaluation_method].append(np.nan)

        # save maximum rec score
        max_score_dict[evaluation_method].append(max(score_rec_list))


        ylabel = 'Mean mutual information' if evaluation_method.startswith('mutual') else 'R2 score'
        plot_scores(score_ppg, score_rec_list, rates_list, ylabel, activity_number, evaluation_method)
        if evaluation_method == 'regression_cv':
            plt.ylim(max(-0.5, min(score_rec_list) - 0.05), max(score_rec_list))
        plt.savefig(target_dir + 'activity_number=' + str(activity_number) + '-' + evaluation_method)
        plt.close()


if plot_bar_charts:
    for evaluation_method in ['mutual_info', 'mutual_info_sklearn', 'regression_insample', 'regression_cv']:
        


        # plot absolute max ADM performance
        plt.bar(activity_range, np.array(max_score_dict[evaluation_method] - np.array(ppg_score_dict[evaluation_method])))
        plt.xlabel('Activity ID')
        plt.ylabel('Absolute similarity score improvement')
        plt.title(evaluation_method + ' - best ADM score improvement')
        plt.gca().xaxis.set_major_locator(mticker.FixedLocator(activity_range))
        plt.savefig(target_dir + 'absolute-improvement' + '-' + evaluation_method)
        plt.close()

        # plot ppg performance
        plt.bar(activity_range, np.array(ppg_score_dict[evaluation_method]))
        plt.xlabel('Activity ID')
        plt.ylabel('PPG Score')
        plt.title(evaluation_method + ' - PPG score')
        plt.gca().xaxis.set_major_locator(mticker.FixedLocator(activity_range))
        plt.savefig(target_dir + 'ppg-score' + '-' + evaluation_method)
        plt.close()

if plot_per_activity_bar_charts:
    
    for activity_idx in activity_range:

        plt.bar(
            ['Raw PPG MI', 'Best ADM MI', 'Raw PPG R2', 'Best ADM R2'],
            [
                ppg_score_dict['mutual_info_sklearn'][activity_idx-1],
                max_score_dict['mutual_info_sklearn'][activity_idx-1],
                ppg_score_dict['regression_cv'][activity_idx-1],
                max_score_dict['regression_cv'][activity_idx-1]
            ]
        )

        plt.title('Activity {}'.format(activity_idx))
        plt.ylabel('Score')
        plt.savefig(target_dir + 'per_activity_bar_charts/activity-' + str(activity_idx))
        plt.close()