import pandas as pd
from config import *
import numpy as np
import pickle
from Helpers.utils_edin import unique
from sklearn import preprocessing
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical

np.random.seed(1337) # for reproducibility

all_full_codes = []
all_50_codes = []
all_clean_codes = []

labels_list = []
labels_50_list = []
labels_clean_list = []
print('Loading the dataset...')

for split in ["train","val","test"]:
    
    with open(MIMIC_3_DIR + "/mimiciii_50_{}.pkl".format(split), "rb") as file:
        data_50_labels = pickle.load(file)
    data_50_labels = data_50_labels["target"].tolist()

    with open(MIMIC_3_DIR + "/mimiciii_clean_{}.pkl".format(split), "rb") as file:
        data_clean_labels = pickle.load(file)
    data_clean_labels = data_clean_labels["target"].tolist()

    print(len(data_clean_labels))
        

# Remove repeated codes
    for i in range(len(data_50_labels)):
        data_50_labels[i] = unique(data_50_labels[i])
    labels_50_list.append(data_50_labels)

    for i in range(len(data_clean_labels)):
        data_clean_labels[i] = unique(data_clean_labels[i])
    labels_clean_list.append(data_clean_labels)
    
    all_50_codes += [x for sublist in data_50_labels for x in sublist] 
    all_clean_codes += [x for sublist in data_clean_labels for x in sublist] 


all_50 = unique(all_50_codes) # 50 unique codes
print("unique MIMIC-III-50 codes: ",len(all_50))

all_clean = unique(all_clean_codes)
print("unique MIMIC-III-clean codes: ",len(all_clean))

print("Load top-50 labelizer ...")
le_50 = np.load(MIMIC_3_DIR + '/le50.npy', allow_pickle=True).item()

print("Load clean labelizer ...")
le_clean = np.load(MIMIC_3_DIR + '/le_clean.npy', allow_pickle=True).item()

print("Transform categorical strings into one hot encodings")
for label_50,label_clean,split in zip(labels_50_list,labels_clean_list,["train","val","test"]):
    print(f"{split} split ...")
    labels_50_int = [x[:] for x in label_50]
    labels_clean_int = [x[:] for x in label_clean]

    for i in range(len(labels_50_int)):
        labels_50_int[i] = list(le_50.transform(labels_50_int[i]))
    for i in range(len(labels_clean_int)):
        labels_clean_int[i] = list(le_clean.transform(labels_clean_int[i]))

    num_50_labels = len(all_50)
    num_clean_labels = len(all_clean)

    labels_50_1hot = np.zeros((len(labels_50_int), num_50_labels), dtype=np.float64)
    print("Top-50 1 hot shape: ",np.shape(labels_50_1hot))
    labels_clean_1hot = np.zeros((len(labels_clean_int), num_clean_labels), dtype=np.float64)
    print("Clean 1 hot shape: ",np.shape(labels_clean_1hot))

    print("Dealing with Top-50 split ...")
    for i in tqdm(range(len(labels_50_int))):
        labels_50_1hot[i,:] = sum(to_categorical(labels_50_int[i],num_50_labels))
    
    print("Dealing with Clean split ...")
    for i in tqdm(range(len(labels_clean_int))):
        labels_clean_1hot[i,:] = sum(to_categorical(labels_clean_int[i],num_clean_labels))
    
    np.savez_compressed(MIMIC_3_DIR + f'/{split}50_1hot.npz', labels_50_1hot)
    np.savez_compressed(MIMIC_3_DIR + f'/{split}clean_1hot.npz', labels_clean_1hot)