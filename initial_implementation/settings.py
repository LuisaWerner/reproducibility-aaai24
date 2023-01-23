VALIDATION_PERCENTAGE = 0.2
EPOCHS = 300
EPOCHS_KENN = 300
ES_ENABLED = False
ES_MIN_DELTA = 0.001
ES_PATIENCE = 30
RANDOM_SEED = 0
RUNS = 30


class Cora:
    num_features = 1433
    num_classes = 7


class CiteSeer:
    num_features = 3703
    num_classes = 6


class PubMed:
    num_features = 500
    num_classes = 3



class Params:
    def __init__(self, name, num_features, num_classes):
        self.DATASET = name
        self.NUMBER_OF_FEATURES = num_features
        self.NUMBER_OF_CLASSES = num_classes







