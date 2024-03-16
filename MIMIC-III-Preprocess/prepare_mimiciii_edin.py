import logging
from functools import partial
from pathlib import Path
import random
import pandas as pd

from Helpers.utils_edin import *
from Helpers.helper_funcs import *
from Helpers.stratify import *
from config import *
import pickle

random.seed(10)
data_dir = Path(DOWNLOAD_DIRECTORY)

preprocessor = TextPreprocessor(
    lower=True,
    remove_special_characters_mullenbach=True,
    remove_special_characters=False,
    remove_digits=True,
    remove_accents=False,
    remove_brackets=False,
    convert_danish_characters=False,
)

get_duplicated_icd9_proc_codes()
# MIMIC-III full
mimic_notes = get_mimiciii_notes(data_dir)
discharge_summaries = prepare_discharge_summaries(mimic_notes)
merged_codes = download_and_preprocess_code_systems(CODE_SYSTEMS)

full_dataset = discharge_summaries.merge(
    merged_codes, on=[SUBJECT_ID_COLUMN, ID_COLUMN], how="inner"
)
full_dataset = replace_nans_with_empty_lists(full_dataset)
# Remove codes that appear less than 10 times
full_dataset = filter_codes(
    full_dataset, [TARGET_COLUMN, "icd9_proc", "icd9_diag"], min_count=MIN_TARGET_COUNT
)
# Remove admissions with no codes
full_dataset = full_dataset[full_dataset[TARGET_COLUMN].apply(len) > 0]

full_dataset = preprocess_documents(df=full_dataset, preprocessor=preprocessor)

print(f"{full_dataset[ID_COLUMN].nunique()} number of admissions")
full_dataset = full_dataset.reset_index(drop=True)


mimic_clean = full_dataset
mimic_clean[TARGET_COLUMN] = mimic_clean[TARGET_COLUMN].apply(lambda x: list(x))

# Generate splits

splits = mimic_clean[[SUBJECT_ID_COLUMN, ID_COLUMN]]
subject_series = mimic_clean.groupby(SUBJECT_ID_COLUMN)[TARGET_COLUMN].sum()
subject_ids = subject_series.index.to_list()
codes = subject_series.to_list()
subject_ids_train, subject_ids_test = iterative_stratification(
    subject_ids, codes, [1 - TEST_SIZE, TEST_SIZE]
)
codes_train = [codes[subject_ids.index(subject_id)] for subject_id in subject_ids_train]
val_size = VAL_SIZE / (1 - TEST_SIZE)
subject_ids_train, subject_ids_val = iterative_stratification(
    subject_ids_train, codes_train, [1 - val_size, val_size]
)

codes_train = [codes[subject_ids.index(subject_id)] for subject_id in subject_ids_train]
codes_val = [codes[subject_ids.index(subject_id)] for subject_id in subject_ids_val]
codes_test = [codes[subject_ids.index(subject_id)] for subject_id in subject_ids_test]

splits.loc[splits[SUBJECT_ID_COLUMN].isin(subject_ids_train), "split"] = "train"
splits.loc[splits[SUBJECT_ID_COLUMN].isin(subject_ids_val), "split"] = "val"
splits.loc[splits[SUBJECT_ID_COLUMN].isin(subject_ids_test), "split"] = "test"

print("------------- Splits Statistics -------------")
print(
    f"Labels missing in the test set: {labels_not_in_split(codes, codes_test)}"
)
print(
    f"Labels missing in the val set: {labels_not_in_split(codes, codes_val)} %"
)
print(
    f"Labels missing in the train set: {labels_not_in_split(codes, codes_train)} %"
)
print(f"Test: KL divergence: {kl_divergence(codes, codes_test)}")
print(f"Val: KL divergence: {kl_divergence(codes, codes_val)}")
print(f"Train: KL divergence: {kl_divergence(codes, codes_train)}")
print(f"Test Size: {len(codes_test) / len(codes)}")
print(f"Val Size: {len(codes_val) / len(codes)}")
print(f"Train Size: {len(codes_train) / len(codes)}")

splits = splits[[ID_COLUMN, "split"]].reset_index(drop=True)

splits.set_index("_id",inplace=True)
mimic_clean.set_index("_id", inplace=True)
mimic_clean["split"] = splits
mimic_clean.reset_index(inplace=True)
mimic_clean.to_feather(data_dir / "mimiciii_clean.feather")

with open(MIMIC_3_DIR + "/mimiciii_clean_train.pkl", "wb") as file:
    pickle.dump(mimic_clean[mimic_clean["split"] == "train"], file)

with open(MIMIC_3_DIR + "/mimiciii_clean_val.pkl", "wb") as file:
    pickle.dump(mimic_clean[mimic_clean["split"] == "val"], file)

with open(MIMIC_3_DIR + "/mimiciii_clean_test.pkl", "wb") as file:
    pickle.dump(mimic_clean[mimic_clean["split"] == "test"], file)