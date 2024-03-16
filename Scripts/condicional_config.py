from config import *
if FILE_NAME == "":
    print("ERROR: FILE_NAME is empty")
    exit(1)
if MODE not in ["train","test"]:
    print("ERROR: Invalid MODE = {}. You must select a valid mode from the following list {}".format(MODE,["train","test"]))
    exit(1)
if MODEL_TYPE not in ["CE","LE","CE_MSAM","LE_MSAM","CE_MSAM_CLQ","LE_MSAM_CLQ"]:
    print("ERROR: Invalid MODEL_TYPE = {}. You must select a valid model type from the following list {}".format(MODEL_TYPE,["CE","LE","CE_MSAM","LE_MSAM"]))
    exit(1)
if DATA_TYPE not in ["top50", "clean"]:
    print("ERROR: Invalid DATA_TYPE = {}. You must select a valid data type from the following list {}".format(DATA_TYPE,["top50", "clean"]))
    exit(1)
if "MSAM" in MODEL_TYPE:
    if SELECTION_CRITERION not in ["rand","MDP"]:
        print("ERROR: Invalid SELECTION_CRITERION = {}. You must select a valid selection criterion from the following list {}".format(SELECTION_CRITERION,["rand","MDP"]))
        print("We advise you to select the synonym selection criterion of Maximum Diversity Problem (MDP)")
        exit(1)
if SELECTION_CRITERION == "rand":
    print("Warning: You selected SELECTION_CRITERION = rand, this indicate that a random synonym selection criterion.")
    print("Warning: We advise you to select the Maximum Diversity Problem (MDP) synonym selection criterion")


if "CLQ" not in MODEL_TYPE:
    MODEL_FROM_CHECKPOINT = START_MODEL_FROM_CHECKPOINT
else:
    MODEL_FROM_CHECKPOINT = START_CLQ_FROM_CLASSIFIER_CHECKPOINT

#Tokenizer Arguments
if ("CE" in MODEL_TYPE) & (DATA_TYPE == "top50"):
    MAX_TEXT_LENGTH = 7142 #8192 or 7142 or 6122
elif ("CE" in MODEL_TYPE) & (DATA_TYPE == "clean"):
    MAX_TEXT_LENGTH = 6122 #8192 or 7142 or 6122
elif "LE" in MODEL_TYPE:
    MAX_TEXT_LENGTH = 8192
MIN_TEXT_LENGTH = 512

# CLQ Arguments
if DATA_TYPE == "top50":
    MLP_HIDDEN_SIZE = 32
elif DATA_TYPE == "clean":
    MLP_HIDDEN_SIZE = 3072

# Chunk Encoder Arguments
OVERLAP_WINDOW = 255

# TRAINING ARGUMENTS
GROUP_BY_LENGTH = True
EPOCHS = 300
LEARNING_RATE = 2e-5
GRADIENT_ACCUMULATION_STEPS = 16
LR_SCHEDULER_TYPE = "linear"
EVALUATION_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
LOGGING_STRATEGY = "epoch"
SAVE_TOTAL_LIMIT = 20
LOAD_BEST_MODEL_AT_END = True
if "CLQ" in MODEL_TYPE:
    METRIC_FOR_BEST_MODEL = "mece"
    GREATER_IS_BETTER = False
else:
    METRIC_FOR_BEST_MODEL = "f1_micro"
    GREATER_IS_BETTER = True
OPTIM = "adamw_torch"
EARLY_STOPPING_PATIENCE = 5