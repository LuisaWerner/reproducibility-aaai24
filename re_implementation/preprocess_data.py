import torch
import torch_geometric.datasets
from torch_geometric.data import Data
import Transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
import numpy as np
import pandas as pd
from pathlib import Path


class PygDataset:
    """ loades and preprocesses the dataset specified in the conf file """

    def __init__(self, args):
        if args.dataset == 'CiteSeer_reproduce':
            _dataset = CiteSeer_reproduce(args).citeseer
        elif args.dataset == 'Cora_reproduce':
            _dataset = Cora_reproduce(args).cora
        elif args.dataset == 'PubMed_reproduce':
            _dataset = PubMed_reproduce(args).pubmed
        else:
            raise ValueError(
                f'Unknown dataset {args.dataset} specified. Use one out of: Cora-reproduce, CiteSeer-preproduce, '
                f'PubMed-reproduce')

        [self._data] = _dataset
        self._data.name = args.dataset

    @property
    def data(self):
        return self._data


class CiteSeer_reproduce:
    def __init__(self, args):
        self.training_dimension = args.training_dimension
        self.data_path = Path(__file__).parent.parent / 'CiteSeer_reproduce'
        self.content = pd.read_csv(self.data_path / 'CiteSeer.content', delimiter='\t', header=None)
        self.relations = pd.read_csv(self.data_path / 'CiteSeer.cites', delimiter='\t', header=None).to_numpy()
        self.num_classes = 6
        self.valid_dim = args.valid_dim
        self.transductive = True if args.mode == 'transductive' else False

        ids, x, y = np.split(self.content, [1, -1], axis=1)
        self.num_nodes = x.shape[0]
        train_len, valid_len = self.get_train_and_valid_lengths(x)

        train_mask = torch.from_numpy(
            np.array([True if x in range(train_len) else False for x in range(self.num_nodes)]))
        val_mask = torch.from_numpy(
            np.array([True if x in range(train_len, train_len + valid_len) else False for x in range(self.num_nodes)]))
        test_mask = torch.from_numpy(np.array(
            [True if x in range(train_len + valid_len, self.num_nodes) else False for x in range(self.num_nodes)]))

        # balances and shuffles ids and x
        p = permute(y.iloc[:, 0], int(train_len // self.num_classes))
        ids = ids.to_numpy().astype(str)[p, :]
        x = x.to_numpy().astype(float)[p, :]
        y = pd.get_dummies(y, prefix=['class']).to_numpy().astype(np.float32)[p, :]
        y = np.argmax(y, axis=1)

        s1, s2 = generate_indexes_transductive(self.relations, ids)
        s1, s2 = torch.LongTensor(s1), torch.LongTensor(s2)
        edge_index = torch.transpose(torch.cat([s1, s2], dim=1), 0, 1)
        self._data = Data(x=torch.Tensor(x), y=torch.squeeze(torch.LongTensor(y)), edge_index=edge_index)
        self._data = T.AddAttributes(args)(self._data)
        self._data.train_mask, self._data.val_mask, self._data.test_mask = train_mask, val_mask, test_mask

    @property
    def citeseer(self):
        return [self._data]

    def get_train_and_valid_lengths(self, features):
        """
        Calculates the numer of samples in training and validation set
        The training set is balanced.
        """
        total_number_of_samples = features.shape[0]
        t_all = int(round(total_number_of_samples * self.training_dimension))
        number_of_samples_training = int(round(t_all * (1. - self.valid_dim)))

        samples_per_class = int(round(number_of_samples_training / self.num_classes))
        number_of_samples_training = samples_per_class * self.num_classes
        number_of_samples_validation = t_all - number_of_samples_training

        return number_of_samples_training, number_of_samples_validation


class Cora_reproduce:
    def __init__(self, args):
        self.valid_dim = args.valid_dim
        self.transductive = True if args.mode == 'transductive' else False
        self.num_classes = 7
        self.training_dimension = args.training_dimension
        self.data_path = Path(__file__).parent.parent / 'Cora_reproduce'
        self.content = pd.read_csv(self.data_path / 'Cora.content', delimiter='\t', header=None)
        ids, x, _ = np.split(self.content, [1, -1], axis=1)
        y = pd.read_pickle(self.data_path / 'Cora_y')
        train_len, valid_len = self.get_train_and_valid_lengths(x)
        p = permute(y.iloc[:, 0], int(train_len // self.num_classes))
        p_swap = {v: k for k, v in dict(list(enumerate(p))).items()}

        # balances and shuffles ids and x
        x = x.to_numpy().astype(float)[p, :]
        y = y.to_numpy().astype(float)[p, :]

        # load indices
        self.relations = pd.read_csv(self.data_path / 'Cora.cites', delimiter='\t', header=None).to_numpy().astype(int)
        s1 = [p_swap[x] for x in self.relations[:, 0]]
        s2 = [p_swap[x] for x in self.relations[:, 1]]
        s1 = torch.unsqueeze(torch.LongTensor(s1), 1)
        s2 = torch.unsqueeze(torch.LongTensor(s2), 1)
        edge_index = torch.transpose(torch.cat([s1, s2], dim=1), 0, 1)

        self.num_nodes = x.shape[0]
        train_mask = torch.from_numpy(
            np.array([True if x in range(train_len) else False for x in range(self.num_nodes)]))
        val_mask = torch.from_numpy(
            np.array([True if x in range(train_len, train_len + valid_len) else False for x in range(self.num_nodes)]))
        test_mask = torch.from_numpy(np.array(
            [True if x in range(train_len + valid_len, self.num_nodes) else False for x in range(self.num_nodes)]))
        train_mask, val_mask, test_mask = train_mask[p], val_mask[p], test_mask[p]

        self._data = Data(x=torch.Tensor(x), y=torch.squeeze(torch.LongTensor(y)), edge_index=edge_index)
        self._data = T.AddAttributes(args)(self._data)
        self._data.train_mask, self._data.val_mask, self._data.test_mask = train_mask, val_mask, test_mask

    @property
    def cora(self):
        return [self._data]

    def get_train_and_valid_lengths(self, features):
        """
        Calculates the numer of samples in training and validation set
        The training set is balanced.
        """
        total_number_of_samples = features.shape[0]
        t_all = int(round(total_number_of_samples * self.training_dimension))
        number_of_samples_training = int(round(t_all * (1. - self.valid_dim)))
        samples_per_class = int(round(number_of_samples_training / self.num_classes))
        number_of_samples_training = samples_per_class * self.num_classes
        number_of_samples_validation = t_all - number_of_samples_training

        return number_of_samples_training, number_of_samples_validation


class PubMed_reproduce:
    def __init__(self, args):
        self.valid_dim = args.valid_dim
        self.transductive = True if args.mode == 'transductive' else False
        self.num_classes = 3
        self.training_dimension = args.training_dimension
        self.data_path = Path(__file__).parent.parent / 'PubMed_reproduce'
        self.content = pd.read_csv(self.data_path / 'PubMed.content', delimiter='\t', header=None)
        ids, x, _ = np.split(self.content, [1, -1], axis=1)
        y = pd.read_pickle(self.data_path / 'PubMed_y')
        train_len, valid_len = self.get_train_and_valid_lengths(x)
        p = permute(y.iloc[:, 0], int(train_len // self.num_classes))
        p_swap = {v: k for k, v in dict(list(enumerate(p))).items()}

        # balances and shuffles ids and x
        x = x.to_numpy().astype(float)[p, :]
        y = y.to_numpy().astype(float)[p, :]

        self.relations = pd.read_csv(self.data_path / 'PubMed.cites', delimiter='\t', header=None).to_numpy().astype(
            int)
        s1 = [p_swap[x] for x in self.relations[:, 0]]
        s2 = [p_swap[x] for x in self.relations[:, 1]]
        s1 = torch.unsqueeze(torch.LongTensor(s1), 1)
        s2 = torch.unsqueeze(torch.LongTensor(s2), 1)
        edge_index = torch.transpose(torch.cat([s1, s2], dim=1), 0, 1)

        self.num_nodes = x.shape[0]
        train_mask = torch.from_numpy(
            np.array([True if x in range(train_len) else False for x in range(self.num_nodes)]))
        val_mask = torch.from_numpy(
            np.array([True if x in range(train_len, train_len + valid_len) else False for x in range(self.num_nodes)]))
        test_mask = torch.from_numpy(np.array(
            [True if x in range(train_len + valid_len, self.num_nodes) else False for x in range(self.num_nodes)]))

        self._data = Data(x=torch.Tensor(x), y=torch.squeeze(torch.LongTensor(y)), edge_index=edge_index)
        self._data = T.AddAttributes(args)(self._data)
        self._data.train_mask, self._data.val_mask, self._data.test_mask = train_mask, val_mask, test_mask

    @property
    def pubmed(self):
        return [self._data]

    def get_train_and_valid_lengths(self, features):
        """
        Calculates the numer of samples in training and validation set
        The training set is balanced.
        """
        total_number_of_samples = features.shape[0]
        t_all = int(round(total_number_of_samples * self.training_dimension))
        number_of_samples_training = int(round(t_all * (1. - self.valid_dim)))
        samples_per_class = int(round(number_of_samples_training / self.num_classes))
        number_of_samples_training = samples_per_class * self.num_classes
        number_of_samples_validation = t_all - number_of_samples_training
        return number_of_samples_training, number_of_samples_validation


def unary_index(ids_list, _id):
    """
    Taken from original implementation
    Returns the index of the id inside ids_list. If multiple indexes are found, throws an error,
    and if no index is found, raises an Exception. If only one is found, returns it.

    Parameters:
    - ids_list = the list of all the ids;
    - id = the specific id we want to know the index of."""
    match = np.where(ids_list == _id)[0]

    assert len(match) < 2

    if len(match) == 0:
        raise Exception(_id)
    else:
        return match[0]


def generate_indexes_transductive(relations, ids_list, verbose=False):
    """
    Taken from original implementation
    Generate the indexes to be used by kenn for the relational part, for the Transductive learning task.
    Specifically, this function returns the index couples (s1,s2) of all the edges for the transductive case:
    i.e. we don't remove edges (n1,n2) s.t. n1 is in the training set, and n2 is in the test set.

    Parameters:
    - relations: np.array containing the edges of the graph. The format is [cited_paper, citing paper];
    - ids_list: np.array containing the ids of all the samples"""
    s1 = []
    s2 = []

    for i in range(len(relations)):
        try:
            match1 = unary_index(ids_list, relations[i, 0])
            match2 = unary_index(ids_list, relations[i, 1])

            s1.append([match1])
            s2.append([match2])
        except Exception as e:
            if verbose:
                print('Missing paper, removed citation! Id: ')
                print(e)
    return np.array(s1).astype(np.int32), np.array(s2).astype(np.int32)


def permute(y, samples_per_class):
    """
    Taken from original implementation
    Return a permutation of the dataset indexes, with samples_per_class samples for each class in the first rows.
    The first rows are used outside as a training set (which must be balanced)

    Parameters:
    - y = the pd.Series containing all the classes of all the samples
    - samples_per_class = int representing the number of samples we want to keep for each class when
      balancing the dataset.
    """
    # List of classes
    classes = list(set(y))

    # initialize empty arrays, will be filled below
    p_train = np.array([], dtype=np.int32)
    p_other = np.array([], dtype=np.int32)

    for c in classes:
        i_c = np.where(y == c)[0]
        p = np.random.permutation(len(i_c))
        i_c = i_c[p]
        p_train = np.concatenate((p_train, i_c[:samples_per_class]), axis=0)
        p_other = np.concatenate((p_other, i_c[samples_per_class:]), axis=0)

    p_train = np.random.permutation(p_train)
    return np.concatenate((p_train, np.random.permutation(p_other)), axis=0)
