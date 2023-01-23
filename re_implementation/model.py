""" Base Neural Network and Knowledge Enhanced Models """
from abc import abstractmethod
import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from kenn.parsers import *
import importlib
import sys, inspect
from preprocess_data import PygDataset
from knowledge import KnowledgeGenerator


def get_model(args):
    """ instantiates the model specified in args """

    msg = f'{args.model} is not implemented. Choose a model in the list: ' \
          f'{[x[0] for x in inspect.getmembers(sys.modules["model"], lambda c: inspect.isclass(c) and c.__module__ == get_model.__module__)]}'
    module = importlib.import_module("model")
    try:
        _class = getattr(module, args.model)
    except AttributeError:
        raise NotImplementedError(msg)

    return _class(args)


class _GraphSampling(torch.nn.Module):
    """
    Super Class for all models' shared components
    """

    def __init__(self, args):
        super(_GraphSampling, self).__init__()
        self.data = PygDataset(args).data
        self.hidden_channels = args.hidden_channels
        self.num_features = self.data.num_features
        self.out_channels = self.data.num_classes
        self.num_layers = args.num_layers
        self.dropout = args.dropout

    def __new__(cls, *args, **kwargs):
        """ avoid instantiation without subclass """
        if cls is _GraphSampling:
            raise TypeError(f'{cls.__name__} can only be baseclass and must not be instantiated without subclass.')
        return super().__new__(cls)

    @abstractmethod
    def forward(self, **kwargs):
        pass

    @abstractmethod
    def reset_parameters(self, **kwargs):
        pass


class Standard(_GraphSampling):
    """ Base Neural Network  """

    def __init__(self, args, **kwargs):
        super(Standard, self).__init__(args)
        self.name = 'Standard'
        self.in_channels = self.data.num_features
        self.lin_layers = ModuleList()
        self.lin_layers.append(Linear(self.in_channels, 50))
        self.lin_layers.append(Linear(50, 50))
        self.lin_layers.append(Linear(50, 50))
        self.lin_layers.append(Linear(50, self.out_channels))
        for lin in self.lin_layers:
            torch.nn.init.xavier_uniform(lin.weight)
            torch.nn.init.zeros_(tensor=lin.bias)

    def forward(self, x, edge_index, relations=None, edge_weight=None):
        for i, lin in enumerate(self.lin_layers[:-1]):
            # x = lin(x)
            x = torch.matmul(x, torch.transpose(torch.Tensor(lin.weight), 0, 1)) + lin.bias
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.lin_layers[-1](x)
        x = torch.matmul(x, torch.transpose(torch.Tensor(self.lin_layers[-1].weight), 0, 1)) + self.lin_layers[-1].bias
        return x


class KENN_Standard(Standard):
    """KENN with Base """

    # clause_weight = 0.5

    def __init__(self, args):
        super().__init__(args)
        self.name = str('KENN_' + self.name)
        self.knowledge = KnowledgeGenerator(self, args).knowledge
        self.kenn_layers = ModuleList()

        self.clause_weight = args.clause_weight
        self.min_weight = args.min_weight
        self.max_weight = args.max_weight

        for _ in range(args.num_kenn_layers):
            self.kenn_layers.append(relational_parser(knowledge_file=self.knowledge, min_weight=self.min_weight,
                                                      max_weight=self.max_weight, initial_clause_weight=self.clause_weight))

    def reset_parameters(self):
        super().reset_parameters()
        for layer in self.kenn_layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, relations=None, edge_weight=None):
        z = super().forward(x, edge_index, relations, edge_weight=edge_weight)

        for layer in self.kenn_layers:
            z, _ = layer(unary=z, edge_index=edge_index, binary=relations)
        return z


