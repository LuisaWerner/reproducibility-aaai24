from time import time
import torch_geometric
import wandb
from app_stats import RunStats, ExperimentStats
from model import get_model
from evaluate import Evaluator
from preprocess_data import *
from training_batch import train, test
import pickle
from pathlib import Path
import json
import argparse


class ExperimentConf(object):
    def __init__(self, conf_dict):
        for key, value in conf_dict.items():
            setattr(self, key, value)


def run_experiment(args):
    torch_geometric.seed_everything(args.seed)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f'Cuda available? {torch.cuda.is_available()}, Number of devices: {torch.cuda.device_count()}')
    results = dict.fromkeys(['test_accuracies'])

    print(f'Start {args.mode} Training')
    xp_stats = ExperimentStats()

    test_accuracies = []
    evaluator = Evaluator(args)

    for run in range(args.runs):

        model = get_model(args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2),
                                     eps=args.adam_eps, amsgrad=False)
        criterion = torch.nn.CrossEntropyLoss()

        train_losses, valid_losses, train_accuracies, valid_accuracies, epoch_time = [], [], [], [], []

        for epoch in range(args.epochs):
            start = time()
            t_loss = train(model, optimizer, device, criterion)
            end = time()
            t_accuracy, _ = test(model, criterion, device, evaluator, mask=model.data.train_mask)
            v_accuracy, v_loss = test(model, criterion, device, evaluator, mask=model.data.val_mask)

            train_accuracies += [t_accuracy]
            valid_accuracies += [v_accuracy]
            train_losses += [t_loss]
            valid_losses += [v_loss]
            epoch_time += [end - start]

            if epoch % args.eval_steps == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Training loss: {t_loss:.10f}, '
                      f'Validation loss: {v_loss:.10f} '
                      f'Train Acc: {t_accuracy:.10f}, '
                      f'Valid Acc: {v_accuracy:.10f} ')

            # early stopping
            if args.es_enabled and evaluator.callback_early_stopping(args, valid_accuracies):
                print(f'Early Stopping at epoch {epoch}.')
                break

        test_accuracy, test_loss = test(model, criterion, device, evaluator, mask=model.data.test_mask)
        test_accuracies += [test_accuracy]
        print(f'Test Accuracy: {test_accuracy:.10f}, Training dimension: {args.training_dimension}')
        rs = RunStats(run, train_losses, train_accuracies, valid_losses, valid_accuracies, test_accuracy, epoch_time,
                      test_accuracies)
        xp_stats.add_run(rs)
        print(rs)
        if args.wandb_use:
            wandb.log(rs.to_dict())
            wandb.run.summary["test_accuracies"] = test_accuracies

        # store the results of run
        results['test_accuracies'] = test_accuracies

        pth = Path(__file__).parent / 'results'
        if not pth.exists():
            pth.mkdir()

        results_dir = pth / str('results_' + str(args.dataset) + '_' + str(args.model) + '_' +
                                                   str(args.training_dimension))

        with open(results_dir, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    xp_stats.end_experiment()
    print(xp_stats)
    if args.wandb_use:
        wandb.log(xp_stats.to_dict())


if __name__ == '__main__':
    # Opening conf file
    path = Path.cwd() / 'conf.json'
    with open(path, 'r') as f:
        json_content = json.loads(f.read())
    # run experiments
    for conf in json_content['configs']:
        run_experiment(ExperimentConf(conf))


