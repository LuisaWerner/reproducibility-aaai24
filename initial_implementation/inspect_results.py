from model import *
from pre_elab import *
from evaluation_functions import *
import settings as s
import pickle

if __name__ == '__main__':

    conf = 0.95

    # check if plots directory exists
    plot_path = Path().resolve() / 'plots'
    if not plot_path.exists():
        plot_path.mkdir()

    path = Path().resolve() / 'results'
    print('Analyze results with initial implementation ')
    for name in ['CiteSeer', 'Cora', 'PubMed']:

        print(f'Dataset {name}')
        try:
            with open(path / str('results_transductive_'+name), 'rb') as input:
                history = pickle.load(input)
        except FileNotFoundError:
            print(f'No results file for {name}. Run  might not have finished yet.')
            continue

        print_and_plot_results(
            history,
            plot_title=f"{name}, {conf * 100}% confidence intervals",
            other_deltas='t',
            confidence_level=conf)