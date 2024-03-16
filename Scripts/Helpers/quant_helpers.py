import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from config_quantification import *
from Helpers.model_helpers import MLP
from torch import nn

def define_samples(samples, y,n_classes):
    result = np.zeros((len(samples),n_classes))
    for sample_idx, sample in enumerate(tqdm(samples)):
        sample_size = len(sample)
        arr = np.zeros((n_classes))
        for x in sample:
            arr += y[x]
        result[sample_idx] = arr/sample_size
    return result

# Evaluation metrics 
def AE(list_p, list_p_hat):
    ae_all_samples = []
    for i in range(len(list_p)): #samples
        ae_sample = []
        for j in range(len(list_p[i])): #labels
            p = list_p[i][j]
            p_hat = list_p_hat[i][j]
            ae = abs(p-p_hat)
            ae_sample.append(ae)
        ae_all_samples.append(sum(ae_sample)/len(list_p[i]))
    return sum(ae_all_samples)/len(list_p)

def smoothing(p, epsilon):
    return (epsilon + p)/(2.0*epsilon + 1.0)

def RAE(list_p, list_p_hat, samples):
    rae_all_samples = []
    for i in range(len(list_p)): #samples
        epsilon = (2.0*len(samples[i]))**(-1)
        rae_sample = []
        for j in range(len(list_p[i])): #labels
            p = smoothing(list_p[i][j], epsilon)
            p_hat = smoothing(list_p_hat[i][j], epsilon)
            rae = abs(p-p_hat)/p + abs(p_hat-p)/(1.0-p)
            rae_sample.append(rae)
        rae_all_samples.append(sum(rae_sample)/(2.0*len(list_p[i])))
    return sum(rae_all_samples)/len(list_p)

class QuantDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return torch.Tensor(self.x[index]).to(DEVICE), torch.Tensor(self.y[index]).to(DEVICE)

def test(model, test_loader,n_samples_test, n_classes):
    
    model.to(DEVICE)

    model.eval()                       
    with torch.no_grad():
        out = np.zeros((n_samples_test, n_classes))
        for j, (x,y) in enumerate(test_loader):
            y_hat = model(x)
            out = y_hat.detach().cpu().numpy()        
    return out

def trainer(model, n_classes, train_loader,test_loader,n_samples_test):
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.HuberLoss(reduction="sum",delta=HUBER_DELTA)
    
    min_train_loss = np.inf
    counter = 0

    all_outputs = []
    
    epoch_losses = []
    
    for epoch in range(EPOCHS):
        model.train()
        
        batch_losses = []
        
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
                
            outputs = model(x)
            
            loss = loss_fn(input=outputs, target=y)

            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        
        train_loss = np.mean(batch_losses)
        
        # mudar se test_batch_size != n_samples_test
        model.eval()                       
        with torch.no_grad():
            out = np.zeros((n_samples_test, n_classes))
            for j, (x,y) in enumerate(test_loader):
                y_hat = model(x)
                out = y_hat.detach().cpu().numpy()        
        all_outputs.append(out)


        if train_loss > min_train_loss: 
            counter += 1
            if counter > PATIENCE:
                print('Early stopping!')
                break
            else:
                print('epoch : {}, train loss : {:.4f}'.format(epoch+1, np.mean(batch_losses)))  
                epoch_losses.append(train_loss)
        else:
            counter = 0
            min_train_loss = train_loss
            torch.save(model.state_dict(), OUTPUT_DIR + "/separated-mlp.pth")
            print('epoch : {}, train loss : {:.4f}'.format(epoch+1, np.mean(batch_losses)))
            epoch_losses.append(train_loss)
    
    best_epoch = np.argmin(epoch_losses)
    print('Best epoch : ',best_epoch+1)
    
    return best_epoch, all_outputs