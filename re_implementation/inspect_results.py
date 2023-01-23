
from scipy.stats import ttest_ind
import statistics
import numpy as np
from pathlib import Path
import pickle


# files for pubmed label Alessandro_esdisabled_speedup
path = Path().resolve() / 'results'
datasets = ['CiteSeer', 'Cora', 'PubMed']
training_dims = [0.1, 0.25, 0.5, 0.75, 0.9]

for name in datasets:
    print('--------------------------------------')
    print(f'Dataset {name}')
    for td in training_dims:
        with open(path / str('results_'+name+'_reproduce_KENN_Standard_'+str(td)), 'rb') as input:
            kenn_file = pickle.load(input)
        kenn = kenn_file['test_accuracies']
        with open(path / str('results_'+name+'_reproduce_Standard_'+str(td)), 'rb') as input:
            standard_file = pickle.load(input)
        standard = standard_file['test_accuracies']
        # print('(p-values: independent two-sample ttest, alternative: kenn test accuracy > standard test accuracy)')
        print(f'{name}, td: {td}: {ttest_ind(standard, kenn, alternative="less", equal_var=False)[1]} (p-values: independent two-sample ttest, alternative: kenn test accuracy > standard test accuracy)')
        print(f'{name}, td: {td}:  {round(np.mean(kenn) - np.mean(standard), 4)} (mean kenn - mean standard):')
