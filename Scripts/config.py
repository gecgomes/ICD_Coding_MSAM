import torch 

# ======================= Classification CONFIG ========================================

DEVICE = torch.device('cuda')

MODEL_TYPE = "LE_MSAM" #or "LE" or "CE" or "LE_MSAM" or "CE_MSAM" or "CE_MSAM_CLQ" or "LE_MSAM_CLQ"
DATA_TYPE = "top50" #or "top50" or "clean"
M = 4
SELECTION_CRITERION = "MDP" #or 'MDP' or 'rand'

FILE_NAME = "test"

START_MODEL_FROM_CHECKPOINT = ""

#CLQ Arguments
HUBER_DELTA = 0.5
QUANT_LAMBDA = 100
START_CLQ_FROM_CLASSIFIER_CHECKPOINT = ""
START_CLQ_FROM_QUANTIFIER_CHECKPOINT = ""

#Train/Test switches for the classifiers
MODE = "test" #or "train" or "test" 

# ===================================================================
# DON'T CHANGE:
PRETRAIN_MODEL = "UFNLP/gatortron-base"

DATA_PATH = "../data"
SAVE_METRICS_PATH = "../models/{}/{}/metrics".format(MODEL_TYPE,FILE_NAME)
SAVE_PREDICTION_PATH = "../models/{}/{}/predictions".format(MODEL_TYPE,FILE_NAME)
OUTPUT_DIR= "../models/{}/{}/model".format(MODEL_TYPE,FILE_NAME)