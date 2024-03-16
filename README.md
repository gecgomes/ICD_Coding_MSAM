# ⚕️Accurate and Well-Calibrated ICD Code Assignment Through Attention Over Diverse Label Embeddings

Official source code repository for the EACL2024 article [Accurate and Well-Calibrated ICD Code Assignment Through Attention Over Diverse Label Embeddings](https://arxiv.org/abs/2402.031728)

```bibtex
@article{gomes2024accurate,
  title={Accurate and Well-Calibrated {ICD} Code Assignment Through Attention Over Diverse Label Embeddings},
  author={Gomes, Gon{\c{c}}alo and Coutinho, Isabel and Martins, Bruno},
  journal={arXiv preprint arXiv:2402.03172},
  year={2024}
}
```

## How to reproduce results
### Setup venv environment
1. Under the directory `Environment` Create a venv environment: `python -m venv env`
2. Activate the environment `env`: `source env/bin/activate`
2. Install the packages : `pip install -r requirements.txt`

### Prepare MIMIC-III splits and Labelizers
This code has been developed on MIMIC-III v1.4. 
1. Please note that you need to complete training to acces the data. The training is free, but takes a couple of hours.  - [link to data access](https://physionet.org/content/mimiciii/1.4/)
2. Open the directory `data_aux`. Download the MIMIC-III data inside this directory by doing: `wget -r -N -c -np --user <physionet_user> --ask-password https://physionet.org/files/mimiciii/1.4/` inside the `data_aux` directory.
3. MIMIC-III/1.4 is now downloaded into `data_aux`, and you should have the following path directories: `data_aux/physionet.org/files/mimiciii/1.4`, with all the MIMIC-III files inside it.
4. Now that you download the MIMIC-III data. Lets do the split described in [Mullenbach et al.](https://github.com/jamesmullenbach/caml-mimic) (MIMIC-III-50), and the split described by [Edin et al.](https://github.com/JoakimEdin/medical-coding-reproducibility)
5. Inside the directory `MIMIC-III-Preprocess`, run the code `python prepare_mimiciii_mullenbach.py` (generate the MIMIC-III-50 split), `python prepare_mimiciii_edin.py` (generate the MIMIC-III-clean split), and then `python prepare_labelizers.py` (to prepare the one hot encoding labels for both splits).
6. Now you can see the train/val/test sets, and the one hot encodins for both splits under the directory `data`

### Data Description
After running the previous steps, the directory `data` with the following files:
* `clean_MSAM4-MDP_synonyms.pkl`: this file contains the selection of 4 synonyms per ICD code for the MIMIC-III-clean unique codes through solving the Maximum Diversity Problem (MDP);
* `top50_MSAM(2/4/8)-MDP_synonyms.pkl`: this file contains the selection of 2/4/8 synonyms per ICD code for the MIMIC-III-50 unique codes through solving the Maximum Diversity Problem (MDP);
* `top50_MSAM(1/2/4/8)-rand_synonyms.pkl`: this file contains 1/2/4/8 synonyms per ICD code for the MIMIC-III-50 unique codes through random selection;
* `icd_mimic3_random_sort.json` & `labels_descriptions_wikipedia.txt`: Descriptions and synonyms from UMLS, wikidata and Wikipedia
* `le.npy`, `le50.npy`, `le_clean.npy`: labelizers regarding the codes from MIMIC-III, MIMIC-III-50, MIMIC-III-clean;
* `mimiciii_(50/clean)_(train/val/test).pkl`: (train/val/test) set for MIMIC-III-(50/clean) split
* `(train/val/test)(50/clean)_1hot.npz`: One hot encodings for the labels in (train/val/test) set of MIMIC-III-(50/clean) split. 

### Running experiments
#### Helpers
The directory `Scripts` has a Helpers directory which contains the helper function scripts to help perform the training and evaluation experiments.

#### Config Files
Inside the `Scripts` directory you can see that there are three different config files: `config.py`, `config_quantification.py`, and `condicional_config.py`.

* `config.py`: this file contains the necessary variables to perform both training and test our different classification models, under the different evaluation datasets;
* `config_quantification.py`: this file contains the necessary variables to perform both training and test our different quantifier models, under the different evaluation datasets;
* `condicional_config.py`: this file contains dynamic variables which their values will depend on variables from the previous config files.

#### Models Directory
The `models` directory is the saving folder for both checkpoints, predictions and metric results. During training/test the results while be automatically saved under this directory.

#### Main Script
The file `main.py` contains the main script to perform both training and evaluation of our classification models.

#### Quantification Script
The file `quantification.py` contains the script to perform both training and evaluation of our quantification models.

### Examples:

#### Example1
Imagine that we want to train the model `CE_MSAM` (Chunk-base encoding strategy with the Multi-synonym attention mechanism.), from scratch, under the MIMIC-III-50 split with four synonyms per code selected with the maximum diversity problem criterion

1. Open the `Scripts/config.file`;
2. Change ``MODEL_TYPE` variable to the desired model type: `MODEL_TYPE = "CE_MSAM"`
3. Change ``DATA_TYPE` variable to the desired data type: `DATA_TYPE = "top50"`
4. Change `M` variable to the desired number of synonyms per ICD code: `M = 4`
5. Change `SELECTION_CRITERION` variable to the desired synonym selection criterion: `SELECTION_CRITERION = "MDP"`
6. Change `FILE_NAME` to a name of your choice that best represent your experiment: `FILE_NAME = "example1"`
7. Change `START_MODEL_FROM_CHECKPOINT` to empty string, since you dont want to start the training from a previous checkpoint: `START_MODEL_FROM_CHECKPOINT = ""`
8. Change `MODE` to training mode: `MODE = "train"`

After setting everything in the `Scripts/config.py` file, run the `main.py` script: `CUDA_VISIBLE_DEVICES=0 python main.py`
Automatically a folder named `models/CE_MSAM/example1` will be created with the following branches:
* `models/CE_MSAM/example1/metrics`: folder that will contains all the metrics regarding the training. Each epoch the files there will be updated to contain most recent metric evaluation results;
* `models/CE_MSAM/example1/model`: file containing the different checkpoints generated from the training process;
* `models/CE_MSAM/example1/predictions`: file containing the predictions for both validation and test set when performing the test process;

#### Example2
Imagine that we want to train the model `LE_MSAM_CLQ` (Longformer encoding strategy with the Multi-synonym attention mechanism, and trained with the joint loss.), from scratch, under the MIMIC-III-clean split with four synonyms per code selected with the maximum diversity problem criterion

1. Open the `Scripts/config.file`;
2. Change ``MODEL_TYPE` variable to the desired model type: `MODEL_TYPE = "LE_MSAM_CLQ"`
3. Change ``DATA_TYPE` variable to the desired data type: `DATA_TYPE = "clean"`
4. Change `M` variable to the desired number of synonyms per ICD code: `M = 4`
5. Change `SELECTION_CRITERION` variable to the desired synonym selection criterion: `SELECTION_CRITERION = "MDP"`
6. Change `FILE_NAME` to a name of your choice that best represent our experiment: `FILE_NAME = "example2"`
7. Change `START_MODEL_FROM_CHECKPOINT` to empty string, since we dont want to start the training from a previous checkpoint: `START_MODEL_FROM_CHECKPOINT = ""`
9. Change `HUBER_DELTA` to the desired value: `HUBER_DELTA = 0.5`
10. Change `QUANT_LAMBDA` to the desired value: `QUANT_LAMBDA = 100`
11. Change `START_CLQ_FROM_CLASSIFIER_CHECKPOINT` to the string describing the path for the desired checkpoint of the classifier "LE_MSAM", since we want to start the classifier from a previous checkpoint: `START_CLQ_FROM_CLASSIFIER_CHECKPOINT = "path/to/LE_MSAM/checkpoint"`
12. Change `START_CLQ_FROM_QUANTIFIER_CHECKPOINT` to the string describing the path for the desired checkpoint of the MLP, since we want to start the quantifier from a previous checkpoint: `START_CLQ_FROM_QUANTIFIER_CHECKPOINT = "path/to/LE_MSAM/models/separated-mlp.pth"`
13. Change `MODE` to training mode: `MODE = "train"`

After setting everything in the `Scripts/config.py` file, run the `main.py` script: `CUDA_VISIBLE_DEVICES=0 python main.py`
Automatically a folder named `models/LE_MSAM_CLQ/example2` will be created with the following branches:
* `models/LE_MSAM_CLQ/example2/metrics`: folder that will contains all the metrics regarding the training. Each epoch the files there will be updated to contain most recent metric evaluation results;
* `models/LE_MSAM_CLQ/example2/model`: file containing the different checkpoints generated from the training process. At the end of the training process it will also be saved under this directory the MLP checkpoint trained under the joint loss optimization;
* `models/LE_MSAM_CLQ/example2/predictions`: file containing the predictions for both validation and test set when performing the test process;

#### Example3
Imagine that we want to train the `MLP` separately from scratch based on the prevalence pcc vectors obtained from the probabilistic predictions of the classifier `CE`, under the MIMIC-III-50 split.

First we need to produce the predictions for both the validation and test set of the MIMIC-III-50 split, so we can easily produce the quantification train and tes splits, that will be used latter to train the MLP.
1. Open the `Scripts/config.file`;
2. Change ``MODEL_TYPE` variable to the desired model type: `MODEL_TYPE = "CE"`
3. Change ``DATA_TYPE` variable to the desired data type: `DATA_TYPE = "50"`
4. Change `FILE_NAME` to a name of your choice that best represent our experiment: `FILE_NAME = "example3"`
5. Change `START_MODEL_FROM_CHECKPOINT` to the best performing checkpoint of our desired model: `START_MODEL_FROM_CHECKPOINT = "path/to/CE/checkpoint"`
6. Change `MODE` to training mode: `MODE = "test"`

After setting everything in the `Scripts/config.py` file, run the `main.py` script: `CUDA_VISIBLE_DEVICES=0 python main.py`
Automatically a folder named `models/LE_MSAM_CLQ/example3` will be created with the following branches:
* `models/LE_MSAM_CLQ/example3/predictions`: file containing the predictions and metrics for both validation and test set;

Now that we have the predictions for both validation and test set under the directory `models/LE_MSAM_CLQ/example3/predictions`, lets train the MLP.
1. Open the `Scripts/config_quantification.file`;
2. Change ``MODEL_TYPE` variable to the desired model type: `MODEL_TYPE = "CE"`
3. Change ``DATA_TYPE` variable to the desired data type: `DATA_TYPE = "50"`
4. Change `FILE_NAME` to a name of your choice that best represent our experiment: `FILE_NAME = "example3"`
5. Change `EPOCHS` variable to the maximum desired number of epochs that you want to train the MLP: `EPOCHS = 300` 
6. Change `PATIENCE` variable to the desired early stop patience: `PATIENCE = 5` 
7. Change `LR` variable to the desired learning rate: `LR = 0.0002` 

The quantification metrics will be saved under `models/LE_MSAM_CLQ/example3/metrics/quantification_metrics.txt`; 
The final MLP checkpoint will be saved under `models/LE_MSAM_CLQ/example3/model/separated-mlp.pth`; 
