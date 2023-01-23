import torch
from kenn.KnowledgeEnhancer import KnowledgeEnhancer
from kenn.boost_functions import GodelBoostConormApprox


class Kenn(torch.nn.Module):
    """
    Kenn module (non-relational)
    """

    def __init__(self, predicates: [str],
                 clauses: [str],
                 min_weight=0.0,
                 max_weight=500.0,
                 activation=lambda x: x,
                 initial_clause_weight=0.5,
                 save_training_data=False,
                 boost_function=GodelBoostConormApprox):

        super().__init__()
        self.activation = activation
        self.knowledge_enhancer = KnowledgeEnhancer(
            predicates, clauses, initial_clause_weight, save_training_data, min_weight=min_weight,
            max_weight=max_weight, boost_function=boost_function)

    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, inputs: torch.Tensor) -> (torch.Tensor, [torch.Tensor, torch.Tensor]):
        """Improve the satisfaction level of a set of clauses.
        :param ground_atoms: the tensor containing the pre-activation values of the ground atoms
        :return: final preactivations"""
        deltas, deltas_list = self.knowledge_enhancer(inputs)
        return self.activation(inputs + deltas), deltas_list
