from kenn import Kenn, KnowledgeEnhancer, RelationalKenn
from kenn.boost_functions import GodelBoostConormApprox


def unary_parser(knowledge_file: str, min_weight=0., max_weight=500., activation=lambda x: x, initial_clause_weight=0.5, save_training_data=False,
                 boost_function=GodelBoostConormApprox):
    """
    Takes in input the knowledge file containing only unary clauses and returns a Kenn Layer,
    with input the predicates and clauses found in the knowledge file.
    :param knowledge_file: path of the prior knowledge file
    :param activation
    :param initial_clause_weight
    :param save_training_data
    :param boost_function
    """
    with open(knowledge_file, 'r') as kb_file:
        predicates_string = kb_file.readline()
        kb_file.readline()
        clauses = kb_file.readlines()

    predicates = predicates_string[:-1].split(',')

    return Kenn(predicates, clauses, min_weight, max_weight, activation, initial_clause_weight, save_training_data,
                boost_function=boost_function)


def unary_parser_ke(knowledge_file: str, min_weight=0., max_weight=500., initial_clause_weight=0.5,
                    boost_function=GodelBoostConormApprox):
    """
    Takes in input the knowledge file containing only unary clauses and returns a Knowledge Enhancer layer,
    with input the predicates and clauses found in the knowledge file.
    :param knowledge_file: path of the prior knowledge file;
    :param min_weight: for clause weight clipping
    :param max_weight: for clause weight clipping
    :param initial_clause_weight
    :param boost_function
    """
    with open(knowledge_file, 'r') as kb_file:
        predicates_string = kb_file.readline()
        kb_file.readline()
        clauses = kb_file.readlines()

    predicates = predicates_string[:-1].split(',')

    return KnowledgeEnhancer(predicates, clauses, min_weight=min_weight, max_weight=max_weight,
                             initial_clause_weight=initial_clause_weight, boost_function=boost_function)


def relational_parser(knowledge_file: str, min_weight=0.0, max_weight=500.0, activation=lambda x: x,
                      initial_clause_weight=0.5,
                      boost_function=GodelBoostConormApprox):
    """
    Takes in input the knowledge file containing both unary and binary clauses and returns a RelationalKenn
    Layer, with input the predicates and clauses found in the knowledge file.
    :param knowledge_file: path of the prior knowledge file;
    :param activation: default linear activation
    :param initial_clause_weight
    :param boost_function
    :param min_weight: clause weight clipping
    :param max-weight: clause weight clipping
    """
    with open(knowledge_file, 'r') as kb_file:
        unary_literals_string = kb_file.readline()
        binary_literals_string = kb_file.readline()

        kb_file.readline()
        clauses = kb_file.readlines()

    u_groundings = [u + '(x)' for u in unary_literals_string[:-1].split(',')]
    b_groundings = [u + '(x)' for u in unary_literals_string[:-1].split(',')] + \
                   [u + '(y)' for u in unary_literals_string[:-1].split(',')] + \
                   [b + '(x.y)' for b in binary_literals_string[:-1].split(',')] + \
                   [b + '(y.x)' for b in binary_literals_string[:-1].split(',')]

    unary_clauses = []
    binary_clauses = []

    reading_unary = True
    for clause in clauses:
        if clause[0] == '>':
            reading_unary = False
            continue

        if reading_unary:
            unary_clauses.append(clause)
        else:
            binary_clauses.append(clause)

    return RelationalKenn(
        unary_predicates=u_groundings,
        binary_predicates=b_groundings,
        unary_clauses=unary_clauses,
        binary_clauses=binary_clauses,
        activation=activation,
        initial_clause_weight=initial_clause_weight,
        min_weight=min_weight,
        max_weight=max_weight,
        boost_function=boost_function)


