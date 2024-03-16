import torch

DEVICE = torch.device("cpu")

# =============== MIMIC-III-50/full settings ============
PAD_CHAR = "**PAD**" #Don't change
EMBEDDING_SIZE = 100 #Don't change
MAX_LENGTH = 2500 #Don't change

# =============== MIMIC-III-Clean settings ==============
PAD_TOKEN = "<PAD>" #Don't change
UNKNOWN_TOKEN = "<UNK>" #Don't change
MIN_TARGET_COUNT = 10 #Don't change
CODE_SYSTEMS = [
    ("ICD9-DIAG", "DIAGNOSES_ICD.csv.gz", "ICD9_CODE", "icd9_diag"),
    ("ICD9-PROC", "PROCEDURES_ICD.csv.gz", "ICD9_CODE", "icd9_proc"),
] #Don't change

ID_COLUMN = "_id" #Don't change
TEXT_COLUMN = "text" #Don't change
TARGET_COLUMN = "target" #Don't change
SUBJECT_ID_COLUMN = "subject_id" #Don't change

TEST_SIZE = 0.15  # Test split ratios (#Don't change)
VAL_SIZE = 0.1  # Val split ratio (#Don't change)
STEP_SIZE = 0.2  # Step size for the iterative stratification (#Don't change)

#================ Data Path ==============================
MIMIC_3_DIR = r'../data' #CHANGE
DOWNLOAD_DIRECTORY = r'../data_aux/physionet.org/files/mimiciii/1.4' #CHANGE

#================ Maximum Diversity Probem settings ======
M = 4 #CHANGE