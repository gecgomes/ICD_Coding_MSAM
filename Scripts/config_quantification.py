import torch 

# ======================= CONFIG ========================================

DEVICE = torch.device('cuda')

MODEL_TYPE = "CE_MSAM_CLQ" #or "LE" or "CE" or "LE_MSAM" or "CE_MSAM" or "CE_MSAM_CLQ" or "LE_MSAM_CLQ"
DATA_TYPE = "top50" #or "top50" or "clean"
FILE_NAME = "test"

MODE = "test"
EPOCHS = 5
PATIENCE = 5
HUBER_DELTA = 0.5
LR = 0.0002

#DON'T CHANGE
SAVE_METRICS_PATH = "../models/{}/{}/metrics".format(MODEL_TYPE,FILE_NAME)
PATH_TO_PREDICTIONS = "../models/{}/{}/predictions".format(MODEL_TYPE,FILE_NAME)
OUTPUT_DIR= "../models/{}/{}/model".format(MODEL_TYPE,FILE_NAME)