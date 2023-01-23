import torch_geometric
from pathlib import Path
import numpy as np
import pandas as pd


def get_dataset(name):
    """
    loads PubMed and Cora from pytorch geometric
    in directory Cora there are the raw data files
    in Cora_reproduce the data in the format of the original citeseer set
    (same for PubMed)
    """
    pth = Path(__file__).parent
    data_path = pth / str(name + '_reproduce')

    try:
        data_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("Folder is already there")
    else:
        print("Folder was created")

    dataset = torch_geometric.datasets.Planetoid(root=name, name=name, split='full')

    ids = pd.DataFrame(np.arange(dataset.data.x.shape[0], dtype=int))
    x = pd.DataFrame(dataset.data.x.numpy().astype(int))
    y = pd.DataFrame(np.expand_dims(dataset.data.y.numpy().astype(str), axis=1))
    y.to_pickle(str(data_path / str(name + '_y')))

    content = pd.concat([ids, x, y], axis=1, ignore_index=True)
    cites = pd.DataFrame(dataset.data.edge_index.numpy()).T
    content.to_csv(path_or_buf=data_path / str(name + '.content'), sep='\t', header=False, index=False)
    cites.to_csv(path_or_buf=data_path / str(name + '.cites'), sep='\t', header=False, index=False)


def load_sets():
    sets = ['Cora', 'PubMed']
    for _set in sets:
        get_dataset(_set)


if __name__ == '__main__':
    load_sets()