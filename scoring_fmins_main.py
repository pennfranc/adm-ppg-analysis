import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_pickle, unpack_data
from mutual_information import scoring_loop_fmins, plot_scores 

sns.set()

fmins = np.linspace(0., 4, 20)
target_dir = './plots/fmins/'

for subject_idx in range(3, 4):
    
    # track progress
    print('processing subject ' + str(subject_idx))

    # load ppg and hr data
    dataset = load_pickle(subject_idx)
    ppg, _, hr, activity, _= unpack_data(dataset)
    ppg = ppg[:-(64)]


    for evaluation_method in ['mutual_info', 'mutual_info_sklearn', 'regression_insample', 'regression_cv']:
 
        scores = scoring_loop_fmins(
            [ppg], [hr],
            fmins=fmins,
            evaluation_method=evaluation_method,
            n_neighbors=20,
            fmax=4,
            nperseg=512,
            noverlap=384

        )
        ylabel = 'Mean mutual information' if evaluation_method.startswith('mutual') else 'R2 score'
        plt.plot(fmins, scores)
        plt.ylabel(ylabel)
        plt.xlabel('Minimum frequency considered [Hz]')
        plt.title(evaluation_method + ' - Subject {}'.format(subject_idx))
        if evaluation_method == 'regression_cv':
            plt.ylim(max(-0.5, min(scores)), None)
        plt.savefig(target_dir + 'subject_idx=' + str(subject_idx) + '-' + evaluation_method)
        plt.close( )