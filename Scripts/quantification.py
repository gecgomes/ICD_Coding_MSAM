
import numpy as np
import random
import torch
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader
from tqdm import tqdm
from config_quantification import *
from Helpers.quant_helpers import *
from Helpers.model_helpers import MLP

def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

configure_seed(888) # for reproducibility

#%% Load the dataset

y_val_true = np.load(PATH_TO_PREDICTIONS + "/y_val_true.npy")

y_val_pred = np.load(PATH_TO_PREDICTIONS + "/y_val_pred.npy")

y_val_prob = np.load(PATH_TO_PREDICTIONS + "/y_val_prob.npy")


y_test_true = np.load(PATH_TO_PREDICTIONS + "/y_test_true.npy")

y_test_pred = np.load(PATH_TO_PREDICTIONS + "/y_test_pred.npy")

y_test_prob = np.load(PATH_TO_PREDICTIONS + "/y_test_prob.npy")


n_classes = y_val_true.shape[1]

#%%

n_samples_train = 5000

samples_train = []

for i in tqdm(range(n_samples_train)):
    sample_size = random.randint(1, len(y_val_true))
    samples_train.append(random.sample(range(0, len(y_val_true)), sample_size))


#%%

n_samples_test = 1000

samples_test = []

for i in tqdm(range(n_samples_test)):
    sample_size = random.randint(1, len(y_test_true))
    samples_test.append(random.sample(range(0, len(y_test_true)), sample_size))


#%% Define samples

# Train
count_train = define_samples(samples_train, y_val_true,n_classes)
cc_train = define_samples(samples_train, y_val_pred,n_classes)
pcc_train = define_samples(samples_train, y_val_prob,n_classes)


# Test
count_test = define_samples(samples_test, y_test_true,n_classes)
cc_test = define_samples(samples_test, y_test_pred,n_classes)
pcc_test = define_samples(samples_test, y_test_prob,n_classes)


print('TRAINING SET')
ae_cc_train_per_class = [mean_absolute_error(count_train[:,j],cc_train[:,j]) for j in range(n_classes)]
ae_cc_train = sum(ae_cc_train_per_class)/n_classes
rae_cc_train = RAE(count_train, cc_train, samples_train)
print('Classify and count:')
print('AE = '+str(round(ae_cc_train,5)))
print('RAE = '+str(round(rae_cc_train,5)))

ae_pcc_train_per_class = [mean_absolute_error(count_train[:,j],pcc_train[:,j]) for j in range(n_classes)]
ae_pcc_train = sum(ae_pcc_train_per_class)/n_classes
rae_pcc_train = RAE(count_train, pcc_train, samples_train)
print('Probabilistic classify and count:')
print('AE = '+str(round(ae_pcc_train,5)))
print('RAE = '+str(round(rae_pcc_train,5)))


print('TEST SET')
pcc_train_mean = np.mean(pcc_train,axis=0)
lazy_test =  np.repeat(np.expand_dims(pcc_train_mean, axis=0), n_samples_test, axis=0)
ae_lazy_test = AE(count_test, lazy_test)
rae_lazy_test = RAE(count_test, lazy_test, samples_test)
print('Lazy:')
print('AE = '+str(round(ae_lazy_test,10)))
print('RAE = '+str(round(rae_lazy_test,10)))

ae_cc_test_per_class = [mean_absolute_error(count_test[:,j],cc_test[:,j]) for j in range(n_classes)]
ae_cc_test = sum(ae_cc_test_per_class)/n_classes
rae_cc_test = RAE(count_test, cc_test, samples_test)
print('Classify and count:')
print('AE = '+str(round(ae_cc_test,10)))
print('RAE = '+str(round(rae_cc_test,10)))

ae_pcc_test_per_class = [mean_absolute_error(count_test[:,j],pcc_test[:,j]) for j in range(n_classes)]
ae_pcc_test = sum(ae_pcc_test_per_class)/n_classes
rae_pcc_test = RAE(count_test, pcc_test, samples_test)
print('Probabilistic classify and count:')
print('AE = '+str(round(ae_pcc_test,10)))
print('RAE = '+str(round(rae_pcc_test,10)))

    
train_dataset = QuantDataset(x=pcc_train, y=count_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

test_dataset = QuantDataset(x=pcc_test, y=count_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=n_samples_test, shuffle=False)


#%%

torch.manual_seed(0)
if DATA_TYPE == "top50":
    hidden_size = 32
else:
    hidden_size = 3072

n_classes = len(y_test_true[0])

if MODE == "test":
    model = MLP(hidden_size=hidden_size,n_classes=n_classes)
    if "CLQ" in MODEL_TYPE:
        model.load_state_dict(torch.load(OUTPUT_DIR + "/clq-mlp.pth"))
    else:
        model.load_state_dict(torch.load(OUTPUT_DIR + "/separated-mlp.pth"))
    count_test_final = test(model,test_loader,n_samples_test, n_classes)
    count_test_final = np.array(count_test_final)
else:
    model = MLP(hidden_size=hidden_size,n_classes=n_classes)
    model.to(DEVICE)

    best_epoch, all_outputs = trainer(model,LR, EPOCHS, PATIENCE, n_classes,train_loader,test_loader,n_samples_test)
    count_test_final = all_outputs[best_epoch]
    count_test_final = np.array(count_test_final)

print(np.shape(count_test))
print(np.shape(count_test_final))
ae_test_final_per_class = [mean_absolute_error(count_test[:,j],count_test_final[:,j]) for j in range(n_classes)]
ae_test_final = sum(ae_test_final_per_class)/n_classes
rae_test_final = RAE(count_test, count_test_final, samples_test)

print('AE = ', round(ae_test_final,10))
print('RAE = ', round(rae_test_final,10))

file = open(SAVE_METRICS_PATH + "/quantify_metrics.txt", "w")
file.write("Classify and count: \n AE: {} \n RAE: {} \n\n Probabilistic classify and count: \n AE: {} \n RAE: {} \n\n  MLP: \n AE: {} \n RAE: {}".format(ae_cc_test,rae_cc_test,ae_pcc_test,rae_pcc_test,ae_test_final,rae_test_final))
file.close()