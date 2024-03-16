import logging
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Optional

import pandas as pd
import wget

from config import ID_COLUMN, SUBJECT_ID_COLUMN, TARGET_COLUMN, TEXT_COLUMN, DOWNLOAD_DIRECTORY

DOWNLOAD_DIRECTORY = Path(DOWNLOAD_DIRECTORY)

def unique(list1): 
    unique_list = [] 
    for x in list1: 
        if x not in unique_list: 
            unique_list.append(x)    
    return unique_list

def make_version_dir(output_dir: Path) -> Path:
    """Creates a new directory for the current version of the dataset."""
    if not output_dir.is_dir():
        output_dir = output_dir / "1"
        logging.info(f"Creating directory {output_dir}")
        output_dir.mkdir(parents=True)
    else:
        logging.info(f"Directory {output_dir} already exists")
        latest_version = max(output_dir.iterdir(), key=lambda x: int(x.name))
        new_version = str(int(latest_version.name) + 1)
        logging.info(
            f"Latest version is {latest_version.name}, bumping to {new_version}"
        )
        output_dir = output_dir / new_version
        logging.info(f"Creating directory {output_dir}")
        output_dir.mkdir()
    return output_dir


def get_mimiciii_notes(download_dir: Path) -> pd.DataFrame:
    """Reads the notes from the mimiciii dataset into a dataframe."""
    return load_gz_file_into_df(download_dir / "NOTEEVENTS.feather")


def load_gz_file_into_df(path: Path, dtype: Optional[dict] = None):
    """Reads the notes from a path into a dataframe. Saves the file as a feather file."""
    download_dir = path.parents[0]
    stemmed_filename = path.name.split(".")[0]
    if (download_dir / f"{stemmed_filename}.feather").is_file():
        logging.info(
            f"{stemmed_filename}.feather already exists, loading data from {stemmed_filename}.feather into a pandas dataframe."
        )
        return pd.read_feather(download_dir / f"{stemmed_filename}.feather")

    logging.info(
        f"Loading data from {stemmed_filename}.csv.gz into a pandas dataframe. This may take a while..."
    )
    file = pd.read_csv(
        download_dir / f"{stemmed_filename}.csv.gz", compression="gzip", dtype=dtype,low_memory=False,
    )
    file.to_feather(download_dir / f"{stemmed_filename}.feather")

    return file


def download_mullenbach_icd9_description() -> pd.DataFrame:
    """Download the icd9 description file from the mullenbach github repo

    Returns:
        pd.DataFrame: ICD9 description dataframe
    """
    logging.info("Downloading ICD9 description file...")
    url = "https://raw.githubusercontent.com/jamesmullenbach/caml-mimic/master/mimicdata/ICD9_descriptions"
    df = pd.read_csv(url, sep="\t", header=None,low_memory=False)
    df.columns = ["icd9_code", "icd9_description"]
    return df


def get_icd9_descriptions(download_dir: Path) -> pd.DataFrame:
    """Gets the IC  D9 descriptions"""
    icd9_proc_desc = pd.read_csv(
        download_dir / "D_ICD_PROCEDURES.csv.gz",
        compression="gzip",
        dtype={"ICD9_CODE": str},
        low_memory=False,
    )
    icd9_proc_desc = clean_icd9_desc_df(icd9_proc_desc, is_diag=False)
    icd9_diag_desc = pd.read_csv(
        download_dir / "D_ICD_DIAGNOSES.csv.gz",
        compression="gzip",
        dtype={"ICD9_CODE": str},
        low_memory=False,
    )
    icd9_diag_desc = clean_icd9_desc_df(icd9_diag_desc, is_diag=True)
    icd9_mullenbach_desc = download_mullenbach_icd9_description()
    icd9_desc = pd.concat([icd9_proc_desc, icd9_diag_desc, icd9_mullenbach_desc])
    return icd9_desc.drop_duplicates(subset=["icd9_code"])


def clean_icd9_desc_df(icd_desc: pd.DataFrame, is_diag: bool) -> pd.DataFrame:
    """
    Cleans the ICD9 description dataframe.
    Args:
        icd_desc (pd.DataFrame): ICD9 description dataframe to clean

    Returns:
        pd.DataFrame: Clean ICD9 description dataframe
    """
    icd_desc = icd_desc.rename(
        columns={"ICD9_CODE": "icd9_code", "LONG_TITLE": "icd9_description"}
    )
    icd_desc["icd9_code"] = icd_desc["icd9_code"].astype(str)
    icd_desc["icd9_code"] = icd_desc["icd9_code"].apply(
        lambda code: reformat_icd9(code, is_diag)
    )
    return icd_desc[["icd9_code", "icd9_description"]]


def download_mullenbach_splits(splits: list[str], download_directory: Path) -> None:
    """Downloads the Mullenbach splits from the github repository."""
    for split in splits:
        download_url = f"https://raw.githubusercontent.com/jamesmullenbach/caml-mimic/master/mimicdata/mimic3/{split}_hadm_ids.csv"
        logging.info(f"\nDownloading - {split}:")
        wget.download(download_url, str(download_directory))


def get_mullenbach_splits(
    splits: list[str], download_directory: Path
) -> dict[str, pd.DataFrame]:
    """Gets the downloaded Mullenbach splits."""
    splits_dict = {}
    for split in splits:
        logging.info(f"\nLoading - {split}:")
        splits_dict[split] = pd.read_csv(
            download_directory / f"{split}_hadm_ids.csv", header=None,low_memory=False,
        )
        splits_dict[split].columns = [ID_COLUMN]
    return splits_dict


def filter_mullenbach_splits(
    splits: dict[str, pd.DataFrame], dataset: pd.DataFrame
) -> pd.DataFrame:
    """Filters the Mullenbach splits to only include hadm_ids that are available in the full dataset."""
    filtered_splits = {}
    for split, df in splits.items():
        logging.info(f"\nFiltering - {split}:")
        len_df = len(df)
        filtered_splits[split] = df[df[ID_COLUMN].isin(dataset[ID_COLUMN])]
        filtered_splits[split] = filtered_splits[split].reset_index(drop=True)
        logging.info(f"\t{len_df} -> {len(filtered_splits[split])}")
    return filtered_splits


def save_mullenbach_splits(
    splits: dict[str, pd.DataFrame],
    ouput_directory_50: Path,
    ouput_directory_full: Path,
) -> None:
    """Saves the filtered Mullenbach splits to feather files."""
    split_50_list = []
    split_full_list = []
    for split, df in splits.items():
        df["split"] = split.replace("dev", "val").split("_")[0]
        if "50" in split:
            split_50_list.append(df)
        else:
            split_full_list.append(df)
    pd.concat(split_50_list).reset_index(drop=True).to_feather(
        ouput_directory_50 / "mimiciii_50_splits.feather"
    )
    pd.concat(split_full_list).reset_index(drop=True).to_feather(
        ouput_directory_full / "mimiciii_full_splits.feather"
    )


def merge_code_dataframes(code_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Merges all code dataframes into a single dataframe.

    Args:
        code_dfs (list[pd.DataFrame]): List of code dataframes.

    Returns:
        pd.DataFrame: Merged code dataframe.
    """
    merged_codes = code_dfs[0]
    for code_df in code_dfs[1:]:
        merged_codes = merged_codes.merge(
            code_df, how="outer", on=[SUBJECT_ID_COLUMN, ID_COLUMN]
        )
    return merged_codes


def replace_nans_with_empty_lists(
    df: pd.DataFrame, columns: list[str] = ["icd9_diag", "icd9_proc"]
) -> pd.DataFrame:
    """Replaces nans in the columns with empty lists."""
    for column in columns:
        df.loc[df[column].isnull(), column] = df.loc[df[column].isnull(), column].apply(
            lambda x: []
        )
    return df


def reformat_icd(code: str, version: int, is_diag: bool) -> str:
    """format icd code depending on version"""
    if version == 9:
        return reformat_icd9(code, is_diag)
    elif version == 10:
        return reformat_icd10(code, is_diag)
    else:
        raise ValueError("version must be 9 or 10")


def reformat_icd10(code: str, is_diag: bool) -> str:
    """
    Put a period in the right place because the MIMIC-3 data files exclude them.
    Generally, procedure codes have dots after the first two digits,
    while diagnosis codes have dots after the first three digits.
    """
    code = "".join(code.split("."))
    if not is_diag:
        return code
    return code[:3] + "." + code[3:]


def reformat_icd9(code: str, is_diag: bool) -> str:
    """
    Put a period in the right place because the MIMIC-3 data files exclude them.
    Generally, procedure codes have dots after the first two digits,
    while diagnosis codes have dots after the first three digits.
    """
    code = "".join(code.split("."))
    if is_diag:
        if code.startswith("E"):
            if len(code) > 4:
                return code[:4] + "." + code[4:]
        else:
            if len(code) > 3:
                return code[:3] + "." + code[3:]
    else:
        if len(code) > 2:
            return code[:2] + "." + code[2:]
    return code


def reformat_code_dataframe(row: pd.DataFrame, col: str) -> pd.Series:
    """Takes a dataframe and a column name and returns a series with the column name and a list of codes.

    Example:
        Input:

                subject_id  _id     icd9_diag
        608           2   163353     V3001
        609           2   163353      V053
        610           2   163353      V290

        Output:

        icd9_diag    [V053, V290, V3001]

    Args:
        row (pd.DataFrame): Dataframe with a column of codes.
        col (str): column name of the codes.

    Returns:
        pd.Series: Series with the column name and a list of codes.
    """
    return pd.Series({col: row[col].sort_values().tolist()})


def merge_report_addendum_helper_function(row: pd.DataFrame) -> pd.Series:
    """Merges the report and addendum text."""
    dout = dict()
    if len(row) == 1:
        dout["DESCRIPTION"] = row.iloc[0].DESCRIPTION
        dout[TEXT_COLUMN] = row.iloc[0][TEXT_COLUMN]
    else:
        # row = row.sort_values(["DESCRIPTION", "CHARTDATE"], ascending=[False, True])
        dout["DESCRIPTION"] = "+".join(row.DESCRIPTION)
        dout[TEXT_COLUMN] = " ".join(row[TEXT_COLUMN])
    return pd.Series(dout)


def format_code_dataframe(df: pd.DataFrame, col_in: str, col_out: str) -> pd.DataFrame:
    """Formats the code dataframe.

    Args:
        df (pd.DataFrame): The dataframe containing the codes.
        col_in (str): The name of the column containing the codes.
        col_out (str): The new name of the column containing the codes

    Returns:
        pd.DataFrame: The formatted dataframe.
    """
    df = df.rename(
        columns={
            "HADM_ID": ID_COLUMN,
            "SUBJECT_ID": SUBJECT_ID_COLUMN,
            "TEXT": TEXT_COLUMN,
        }
    )
    df = df.sort_values([SUBJECT_ID_COLUMN, ID_COLUMN])
    df[col_in] = df[col_in].astype(str).str.strip()
    df = df[[SUBJECT_ID_COLUMN, ID_COLUMN, col_in]].rename({col_in: col_out}, axis=1)
    # remove codes that are nan
    df = df[df[col_out] != "nan"]
    return (
        df.groupby([SUBJECT_ID_COLUMN, ID_COLUMN])
        .apply(partial(reformat_code_dataframe, col=col_out))
        .reset_index()
    )


def merge_reports_addendum(mimic_notes: pd.DataFrame) -> pd.DataFrame:
    """Merges the reports and addendum into one dataframe.

    Args:
        mimic_notes (pd.DataFrame): The dataframe containing the notes from the mimiciii dataset.

    Returns:
        pd.DataFrame: The dataframe containing the discharge summaries consisting of reports and addendum.
    """
    discharge_summaries = mimic_notes[mimic_notes["CATEGORY"] == "Discharge summary"]
    discharge_summaries[ID_COLUMN] = discharge_summaries[ID_COLUMN].astype(int)
    return (
        discharge_summaries.groupby([SUBJECT_ID_COLUMN, ID_COLUMN])
        .apply(merge_report_addendum_helper_function)
        .reset_index()
    )


def top_k_codes(df: pd.DataFrame, column_names: list[str], k: int) -> set[str]:
    """Get the top k most frequent codes from a dataframe"""
    df = df.copy()
    counter = Counter()
    for col in column_names:
        list(map(counter.update, df[col]))
    top_k = counter.most_common(k)
    return set(map(lambda x: x[0], top_k))


def filter_codes_mullen(
    df: pd.DataFrame, column_names: list[str], codes_to_keep: set[str]
) -> pd.DataFrame:
    """Filter the codes in the dataframe to only keep the desired codes"""
    df = df.copy()
    for col in column_names:
        df[col] = df[col].apply(
            lambda codes: list(filter(lambda x: x in codes_to_keep, codes))
        )
    return df


def remove_duplicated_codes(df: pd.DataFrame, column_names: list[str]) -> pd.DataFrame:
    """Remove duplicated codes in the dataframe"""
    df = df.copy()
    for col in column_names:
        df[col] = df[col].apply(lambda codes: list(set(codes)))
    return df


class TextPreprocessor:
    def __init__(
        self,
        lower: bool = True,
        remove_special_characters_mullenbach: bool = True,
        remove_special_characters: bool = False,
        remove_digits: bool = True,
        remove_accents: bool = False,
        remove_brackets: bool = False,
        convert_danish_characters: bool = False,
    ) -> None:
        self.lower = lower
        self.remove_special_characters_mullenbach = remove_special_characters_mullenbach
        self.remove_digits = remove_digits
        self.remove_accents = remove_accents
        self.remove_special_characters = remove_special_characters
        self.remove_brackets = remove_brackets
        self.convert_danish_characters = convert_danish_characters

    def __call__(self, df):
        if self.lower:
            df[TEXT_COLUMN] = df[TEXT_COLUMN].str.lower()
        if self.convert_danish_characters:
            df[TEXT_COLUMN] = df[TEXT_COLUMN].str.replace("å", "aa", regex=True)
            df[TEXT_COLUMN] = df[TEXT_COLUMN].str.replace("æ", "ae", regex=True)
            df[TEXT_COLUMN] = df[TEXT_COLUMN].str.replace("ø", "oe", regex=True)
        if self.remove_accents:
            df[TEXT_COLUMN] = df[TEXT_COLUMN].str.replace("é|è|ê", "e", regex=True)
            df[TEXT_COLUMN] = df[TEXT_COLUMN].str.replace("á|à|â", "a", regex=True)
            df[TEXT_COLUMN] = df[TEXT_COLUMN].str.replace("ô|ó|ò", "o", regex=True)
        if self.remove_brackets:
            df[TEXT_COLUMN] = df[TEXT_COLUMN].str.replace("\[[^]]*\]", "", regex=True)
        if self.remove_special_characters:
            df[TEXT_COLUMN] = df[TEXT_COLUMN].str.replace("\n|/|-", " ", regex=True)
            df[TEXT_COLUMN] = df[TEXT_COLUMN].str.replace(
                "[^a-zA-Z0-9 ]", "", regex=True
            )
        if self.remove_special_characters_mullenbach:
            df[TEXT_COLUMN] = df[TEXT_COLUMN].str.replace(
                "[^A-Za-z0-9]+", " ", regex=True
            )
        if self.remove_digits:
            df[TEXT_COLUMN] = df[TEXT_COLUMN].str.replace("(\s\d+)+\s", " ", regex=True)

        df[TEXT_COLUMN] = df[TEXT_COLUMN].str.replace("\s+", " ", regex=True)
        df[TEXT_COLUMN] = df[TEXT_COLUMN].str.strip()
        return df


def preprocess_documents(
    df: pd.DataFrame, preprocessor: TextPreprocessor
) -> pd.DataFrame:  
    df = preprocessor(df)
    df["num_words"] = df.text.str.count(" ") + 1
    df["num_targets"] = df[TARGET_COLUMN].apply(len)
    return df


def get_duplicated_icd9_proc_codes() -> set:
    """Get the duplicated ICD9-PROC codes. The codes are duplicated because they are saved as integers,
    removing any zeros at the beginning of the codes. These codes will not be included in the dataset.

    Returns:
        set: The duplicated ICD9-PROC codes
    """
    icd9_proc_codes = pd.read_csv(
        DOWNLOAD_DIRECTORY / "D_ICD_PROCEDURES.csv.gz",
        compression="gzip",
        dtype={"ICD9_CODE": str},
        low_memory=False,
    )
    return set(
        icd9_proc_codes[icd9_proc_codes["ICD9_CODE"].astype(str).duplicated()][
            "ICD9_CODE"
        ]
    )


def prepare_discharge_summaries(mimic_notes: pd.DataFrame) -> pd.DataFrame:
    """Format the notes dataframe into the discharge summaries dataframe

    Args:
        mimic_notes (pd.DataFrame): The notes dataframe

    Returns:
        pd.DataFrame: Formatted discharge summaries dataframe
    """
    mimic_notes = mimic_notes.rename(
        columns={
            "HADM_ID": ID_COLUMN,
            "SUBJECT_ID": SUBJECT_ID_COLUMN,
            "TEXT": TEXT_COLUMN,
        }
    )
    logging.info(f"{mimic_notes[ID_COLUMN].nunique()} number of admissions")
    discharge_summaries = merge_reports_addendum(mimic_notes)
    discharge_summaries = discharge_summaries.sort_values(
        [SUBJECT_ID_COLUMN, ID_COLUMN]
    )

    discharge_summaries = discharge_summaries.reset_index(drop=True)
    logging.info(
        f"{discharge_summaries[SUBJECT_ID_COLUMN].nunique()} subjects, {discharge_summaries[ID_COLUMN].nunique()} admissions"
    )
    return discharge_summaries


def filter_codes(df: pd.DataFrame, columns: list[str], min_count: int) -> pd.DataFrame:
    """Filter the codes dataframe to only include codes that appear at least min_count times

    Args:
        df (pd.DataFrame): The codes dataframe
        col (str): The column name of the codes
        min_count (int): The minimum number of times a code must appear

    Returns:
        pd.DataFrame: The filtered codes dataframe
    """
    for col in columns:
        code_counts = Counter([code for codes in df[col] for code in codes])
        codes_to_keep = set(
            code for code, count in code_counts.items() if count >= min_count
        )
        df[col] = df[col].apply(lambda x: [code for code in x if code in codes_to_keep])
    return df


def download_and_preprocess_code_systems(code_systems: list[tuple]) -> pd.DataFrame:
    """Download and preprocess the code systems dataframe

    Args:
        code_systems (List[tuple]): The code systems to download and preprocess

    Returns:
        pd.DataFrame: The preprocessed code systems dataframe"""
    code_dfs = []
    for name, fname, col_in, col_out in code_systems:
        logging.info(f"Loading {name} codes...")
        df = pd.read_csv(
            DOWNLOAD_DIRECTORY / fname, compression="gzip", dtype={"ICD9_CODE": str},low_memory=False,
        )
        df = format_code_dataframe(df, col_in, col_out)
        df = remove_duplicated_codes(df, [col_out])
        code_dfs.append(df)

    merged_codes = merge_code_dataframes(code_dfs)
    merged_codes = replace_nans_with_empty_lists(merged_codes)
    merged_codes["icd9_diag"] = merged_codes["icd9_diag"].apply(
        lambda codes: list(map(partial(reformat_icd9, is_diag=True), codes))
    )
    merged_codes["icd9_proc"] = merged_codes["icd9_proc"].apply(
        lambda codes: list(map(partial(reformat_icd9, is_diag=False), codes))
    )
    merged_codes[TARGET_COLUMN] = merged_codes["icd9_proc"] + merged_codes["icd9_diag"]
    return merged_codes