import seaborn as sns
import scipy
from scipy.stats import t
sns.set_theme('notebook')
sns.set_style('whitegrid')

import wandb
from pathlib import Path
api = wandb.Api()
import numpy as np
import pickle


def main():
    n_runs = 30
    initial_path = Path().resolve() / 'initial_implementation' / 'results'
    re_path = Path().resolve() / 're_implementation' / 'results'

    datasets = ['CiteSeer', 'Cora', 'PubMed']
    training_dims = [0.1, 0.25, 0.5, 0.75, 0.9]

    for name in datasets:
        print('-------------------------------')
        print(f'Dataset {name}')
        with open(initial_path / str('results_transductive_' + name) , 'rb') as input:
            history = pickle.load(input)

        for td in training_dims:
            with open(re_path / str('results_' + name + '_reproduce_KENN_Standard_' + str(td)), 'rb') as input:
                kenn_re_file = pickle.load(input)
            kenn_re = kenn_re_file['test_accuracies']
            kenn_initial = [history[str(int(td*100))]['KENN'][i]['test_accuracy'].numpy() for i in range(n_runs)]
            # nn_initial = [history[str(int(td*100))]['KENN'][i]['test_accuracy'].numpy() for i in range(n_runs)]

            mean_initial = np.mean(kenn_initial)
            mean_re = np.mean(kenn_re)
            std_initial = np.std(kenn_initial)
            std_re = np.std(kenn_re)
            print(f'td: {td}')
            print(f'Mean (Std) Initial Implementation: {round(mean_initial, 4)} ({round(std_initial, 4)})')
            print(f'Mean (Std) Re-Implementation: {round(mean_re, 4)} ({round(std_re, 4)})')
            print(f'P-value KS Test, two-sided: {scipy.stats.kstest(kenn_re, kenn_initial)[1]}')

if __name__ == '__main__':
    main()

