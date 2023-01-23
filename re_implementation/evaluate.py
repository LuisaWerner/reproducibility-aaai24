import numpy as np
import torch


class Evaluator:

    def __init__(self, args):
        self.name = args.dataset
        self.es_patience = args.es_patience
        self.eval_steps = args.eval_steps
        self.es_min_delta = args.es_min_delta

    def callback_early_stopping(self, args, valid_accuracies):
        """
        Takes as argument the list with all the validation accuracies.
        If patience=k, checks if the mean of the last k accuracies is higher than the mean of the
        previous k accuracies (i.e. we check that we are not overfitting). If not, stops learning.
        """
        # No early stopping for 2*patience epochs
        if len(valid_accuracies) // args.es_patience < 2:
            return False
        # Mean loss for last patience epochs and second-last patience epochs
        mean_previous = np.mean(valid_accuracies[::-1][args.es_patience:2 * args.es_patience])
        mean_recent = np.mean(valid_accuracies[::-1][:args.es_patience])
        delta = mean_recent - mean_previous

        if delta <= args.es_min_delta:
            print(
                "*CB_ES* Validation Accuracy didn't increase in the last %d epochs" % (args.es_patience))
            print("*CB_ES* delta:", delta)
            return True
        else:
            return False

    def eval_acc_kenn(self, y_true, y_pred):
        """ Accuracy function of KENN-Citeseer-Experiments"""
        correctly_classified = y_pred.squeeze() == y_true
        return torch.mean(correctly_classified.float())
