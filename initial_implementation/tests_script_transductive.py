import numpy as np
from pre_elab import generate_dataset
import training_standard as ts
import training_transductive as t
import pickle
import tensorflow as tf
from settings import *
from pathlib import Path


def run_tests_transductive(
        params,
        include_greedy=False,
        include_e2e=True,
        save_results=True,
        custom_training_dimensions=False,
        verbose=True):

    # SET RANDOM SEED for tensorflow and numpy
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    training_dimensions = []
    if not custom_training_dimensions:
        print("No custom training dimensions found.")
        training_dimensions = [0.1, 0.25, 0.5, 0.75, 0.9]
        print("Using default training dimensions: {}".format(training_dimensions))
    else:
        training_dimensions = custom_training_dimensions

    results_e2e = {}
    results_greedy = {}

    for td in training_dimensions:
        td_string = str(int(td * 100))
        print(' ========= Start training (' + td_string + '%)  =========')
        results_e2e.setdefault(td_string, {})
        results_greedy.setdefault(td_string, {})

        # results e2e will be dictionaries like this:
        # {'10' : {'NN': [list of n_runs dictionaries containing all the stats], 'KENN': [same a before]},
        #  '25' : {'NN': [list of n_runs dictionaries containing all the stats], 'KENN': [same a before]},
        #   ...}

        for i in range(RUNS):
            print('Generate new dataset: iteration number ' + str(i))
            generate_dataset(params, td, verbose=False)

            if include_e2e:
                print("--- Starting Base NN Training ---")
                results_e2e[td_string].setdefault('NN', []).append(
                    ts.train_and_evaluate_standard(params, td, verbose=verbose)[3])
                print("--- Starting KENN Transductive Training ---")
                results_e2e[td_string].setdefault('KENN', []).append(
                    t.train_and_evaluate_kenn_transductive(params, td, verbose=verbose))

        if save_results:
            results_path = Path(__file__).parent / 'results'
            if not results_path.exists():
                results_path.mkdir()

            if include_e2e:
                with open(results_path / f'results_transductive_{params.DATASET}', 'wb') as output:
                    pickle.dump(results_e2e, output)

    return results_e2e, results_greedy


if __name__ == "__main__":
    run_tests_transductive(params=Params('Cora', Cora.num_features, Cora.num_classes), include_greedy=False, include_e2e=True, save_results=True)
