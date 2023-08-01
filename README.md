# What Does It Take to Reproduce Experiments? Evidences from the Neuro-Symbolic Domain.

This repository contains our re-implementation of the experiments conducted with [Knowledge Enhanced Neural Networks (KENN)](https://github.com/rmazzier/KENN-Citeseer-Experiments) on the [Citeseer Dataset](https://linqs.soe.ucsc.edu/data), including the re-implementation of KENN in PyTorch and [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric). We also extended the experiments to the datasets Cora and PubMed. 

| Name           | Description                      | #nodes    | #edges      | #features | #Classes | Task                 |
|----------------|----------------------------------|-----------|-------------|-----------|----------|----------------------|
| CiteSeer       | from Planetoid, Citation Network | 3,327     | 9,104       | 3,703     | 6        | Node classification  |
| Cora           | from Planetoid, Citation Network | 2,708     | 10,556      | 1,433     | 7        | Node Classification  |
| PubMed         | from Planetoid, Citation Network | 19,717    | 88,648      | 500       | 3        | Node Classification  |

## Overview of this repository
This repository contains to sub-directories that refer to the experiments conducted with the initial implementation and the re-implementation.  
1. The initial_implementation contains code from initial experiments [here](https://github.com/rmazzier/KENN-Citeseer-Experiments), extended to Cora and PubMed
2. The re_implementation contains code for the experiments based on PyTorch. 

The results of both approaches are stored in the respective '/results' subdirectory. 

### Before running the experiments in both implementations
1. In order to make sure that the right environment is used, the necessary Python packages and their versions are specified in `requirements.txt`. We use Python 3.9. To install them go in the project directory and create a conda environment with the following packages. 
```
pip install -r requirements.txt
``` 
The full list of packages in our environment including the dependencies is specified in `system_packages.txt`. 

2. While the Citeseer dataset used by the initial implementation is already stored in the directory `CiteSeer_reproduce`, the datasets Cora and PubMed have to be loaded from PyTorch Geometric [[Source](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/planetoid.html)] and preprocessed. 
Run the following command from the project directory to get PubMed and Cora. 
```
python create_datasets.py 
```

### To run the initial experiments. 
1. To run the initial experiments with the specified parameters in the paper on all three datasets, run the following command from the project directory: 
```
cd initial_implementation
python test.py 
```

2. To get an overview of the results of the initial implementation. It is important that all experiments are finished before. 
```
cd initial_implementation
python inspect_results.py
```

### To run the replicated experiment.
We use [Weights and Biases](https://wandb.ai/site) (WandB) as experiment tracking tool. The experiments can be run WITHOUT or WITH  the use of WandB.
1. To run the experiments without WandB, run the following command. 

```
cd re-implementation
python train_and_evaluate.py conf.json 
```

(By default, ```"wandb_use" : false``` is set in `re-implementation/conf.json`)  


2. If you want to use weights and biases specify the following parameters in  `re-implementation/conf.json`.
```
"wandb_use" : true,
"wandb_label": "<your label>",
"wandb_project" : "<your project>",
"wandb_entity": "<your entity>"
```

Then use the following command to run the experiments: 
```
cd re-implementation
python run_experiments.py conf.json
```

Interprete the results 
To get an overview of the results of the re-implementation, run the following command
```
cd re_implementation
python inspect_results.py
```

### Comparison between both implementations 
To compare the results of both approaches (comparison type 2 reproducibility), go to the project folder and run
```
python compare_results.py 
```

### Hyperparameters 
In the file `re-implementation/conf.json`, the hyperparameters and settings of the runs are configured and saved. By default, the conf.json file contains parameters mentioned in the paper. The last column indicates whether the parameter can be modified in this implementation and to which values it should be set. 

| Parameter            | description                                                         | default                    | state                                                                                               |
|----------------------|---------------------------------------------------------------------|----------------------------|-----------------------------------------------------------------------------------------------------|
| adam_beta1           | Adam optimizer parameter                                            | 0.9                        | modifiable                                                                                          |
| adam_beta2           | Adam optimizer parameter                                            | 0.999                      | modifiable                                                                                          |
| adam_eps             | Adam optimizer parameter                                            | 1e-07                      | modifiable                                                                                          |
| bias init            | Initialization of bias in NN                                        | "zeroes"                   |                                                                                                     |
| binary_preactivation | constant value for activation of binary predicate                   | 500.0                      | modifiable                                                                                          |
| clause_weight        | initialization of clause weight                                     | 0.5                        | modifiable                                                                                          |
| dataset              | dataset                                                             | CiteSeer_reproduce         | modifiable: [Cora_reproduce, PubMed_reproduce,  CiteSeer_reproduce]                                 |
| device               | gpu device for gpu computing                                        | 1                          | modifiable                                                                                          |
| dropout              | dropout rate                                                        | 0.0                        | modifiable (0.0, 1.0)                                                                               |
| epochs               | number of epochs in training                                        | 300                        | modifiable (1, ...)                                                                                 |
| es_enabled           | early stopping activated flag                                       | false                      | modifiable [true, false]                                                                            |
| es_min_delta         | early stopping delta threshold                                      | 0.001                      | modifiable (0, ...)                                                                                 |
| es_patience          | early stopping patience                                             | 10                         | modifiable (0, ...)                                                                                 |
| eval_steps           | prints each n steps during  training                                | 10                         | modifiable (1, ...)                                                                                 |
| hidden_channels      | hidden layer dimension                                              | 50                         | modifiable (1, ...)                                                                                 |
| loss function        | loss function                                                       | categorical cross-entropy  |                                                                                                     |
| lr                   | learning rate                                                       | 0.001                      | modifiable (0.0, 1.0)                                                                               |
| min_weight           | clause weight clipping  minimum value                               | 0.0                        | modifiable (..., max_weight)                                                                        |
| max_weight           | clause weight clipping  maximum value                               | 500.0                      | modifiable (min_weight, ...)                                                                        |
| mode                 | training mode                                                       | "transductive"             |                                                                                                     |
| model                | standard or KENN_Standard (for base NN Standard)                    | Standard                   | modifiable: [Standard, KENN_Standard]                                                               |
| num_kenn_layers      | number of KENN layers                                               | 3                          | modifiable (0, ...)                                                                                 |
| num_layers           | number of layers of base NN                                         | 3                          | modifiable (1, ...)                                                                                 |
| optimizer            | optimizer                                                           | adam                       |                                                                                                     |
| runs                 | number of runs                                                      | 30                         | modifiable (1, ...)                                                                                 |
| seed                 | random seed                                                         | 0                          | modifiable (0,...)                                                                                  |
| training_dimension   | training dimension                                                  | 0.1                        | modifiable: [0.1, 0.25, 0.5, 0.75, 0.9]                                                             |
| valid_dim            | validation set dimension                                            | 0.2                        | modifiable in: (0.0, 1.0)                                                                           |
| wandb_use            | if weights and biases  should be used                               | false                      | modifiable: [true, false]                                                                           |
| wandb_label          | label for weights and biases                                        | "None"                     | modifiable depending on custom WandB settings                                                       |
| wandb_project        | project name for weights  and biases                                | "None"                     | modifiable depending on custom WandB settings                                                       |
| wandb_entity         | entity name for weights and biases                                  | "None"                     | modifiable depending on custom WandB settings                                                       |
| weight init          | initialization of weights in NN                                     | xavier uniform             |                                                                                                     |
 









 


