from settings import *
from tests_script_transductive import run_tests_transductive

if __name__ == '__main__':

    # Cora
    params_cora = Params('Cora', Cora.num_features, Cora.num_classes)
    run_tests_transductive(params_cora, include_greedy=False, include_e2e=True, save_results=True)

    # CiteSeer
    params_citeseer = Params('CiteSeer', CiteSeer.num_features, CiteSeer.num_classes)
    run_tests_transductive(params_citeseer, include_greedy=False, include_e2e=True, save_results=True)

    # PubMed
    params_pubmed = Params('PubMed', PubMed.num_features, PubMed.num_classes)
    run_tests_transductive(params_pubmed, include_greedy=False, include_e2e=True, save_results=True)
