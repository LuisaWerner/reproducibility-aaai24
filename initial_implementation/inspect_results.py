from model import *
from pre_elab import *
from evaluation_functions import *
import settings as s
import pickle

if __name__ == '__main__':

    conf = 0.95
    # CiteSeer Original
    with open('results/e2e/repeat /results_transductive_100runs_3layers_repeat', 'rb') as input:
        citeseer_original_transductive = pickle.load(input)
    print('CiteSeer Original')
    print_and_plot_results(
        citeseer_original_transductive,
        plot_title=f"Transductive learning, {conf * 100}% confidence intervals",
        other_deltas='t',
        confidence_level=conf)

    # Cora
    # print('Cora')
    # with open('dataset/Cora/results/e2e/results_transductive_30_nabil', 'rb') as input:
    #     cora_transductive = pickle.load(input)
    #
    # print_and_plot_results(
    #     cora_transductive,
    #     plot_title=f"Transductive learning, {conf * 100}% confidence intervals",
    #     other_deltas='t',
    #     confidence_level=conf)
    #
    # # CiteSeer
    # print('CiteSeer PyG')
    # with open('dataset/CiteSeer/results/e2e/results_transductive_30', 'rb') as input:
    #     citeseer_transductive = pickle.load(input)
    #
    # print_and_plot_results(
    #     citeseer_transductive,
    #     plot_title=f"Transductive learning, {conf * 100}% confidence intervals",
    #     other_deltas='t',
    #     confidence_level=conf)

    # PubMed
    # print('PubMed')
    # with open('dataset/PubMed/results/e2e/results_transductive_30_nabil_earlystopping', 'rb') as input:
    #     pubmed_transductive = pickle.load(input)
    #
    # print_and_plot_results(
    #     pubmed_transductive,
    #     plot_title=f"Transductive learning, {conf * 100}% confidence intervals",
    #     other_deltas='t',
    #     confidence_level=conf)

    # print('Flickr')
    # with open('dataset/Flickr/results/e2e/results_transductive_100', 'rb') as input:
    #     flickr_transductive = pickle.load(input)
    # print_stats(flickr_transductive)
    # # print_and_plot_results(
    # #     flickr_transductive,
    # #     plot_title=f"Transductive learning, {conf * 100}% confidence intervals",
    # #     other_deltas='t',
    # #     confidence_level=conf)

