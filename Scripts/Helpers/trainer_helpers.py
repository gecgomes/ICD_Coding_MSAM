import torch 
from torch import nn
from typing import Union, Dict, List, Optional, Tuple, Any
from config import *
from condicional_config import *
from Helpers.main_helpers import *
from transformers import Trainer, TrainerCallback
from Helpers.evaluation import *
from sklearn.metrics import f1_score, recall_score,precision_score,mean_squared_error,mean_absolute_error
from torchmetrics import CalibrationError
mece = CalibrationError(task= "binary",n_bins=20, norm='l1')
import pickle

class LEDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        aux = self.tokenizer(self.texts[idx])
        max_length = compute_max_length(aux)
        encodings = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=max_length)

        item = {k: torch.tensor(v).unsqueeze(0) for k, v in encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class CEDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        aux = self.tokenizer(self.texts[idx])
        max_length = compute_max_length(aux)
        encodings = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=max_length)
        if max_length > 512:
            item = {k: torch.stack([torch.tensor(v)[i+1:i+511] for i in range(0,max_length-(510-OVERLAP_WINDOW + 1),510-OVERLAP_WINDOW)],dim=0) for k, v in encodings.items()}
            last = item["input_ids"][:,-1]
            sep_tokens = torch.ones((item["input_ids"].shape[0],1))*102
            mask_sep = ((last != 0)*(last != 102)).unsqueeze(1)
            sep_tokens = sep_tokens*mask_sep
            cls_tokens = torch.ones((item["input_ids"].shape[0],1))*101
            item["input_ids"] = torch.cat((cls_tokens,item["input_ids"],sep_tokens), dim = 1)
            item["token_type_ids"] = torch.cat((torch.zeros(item["token_type_ids"].shape[0],1),item["token_type_ids"],torch.zeros(item["token_type_ids"].shape[0],1)*mask_sep), dim = 1)
            item["attention_mask"] = torch.cat((torch.ones(item["attention_mask"].shape[0],1),item["attention_mask"],torch.ones(item["attention_mask"].shape[0],1)*mask_sep), dim = 1)
        else:
            item = {k: torch.tensor(v).unsqueeze(0) for k, v in encodings.items()}
        item["input_ids"] = item["input_ids"].type(torch.long)
        item["token_type_ids"] = item["token_type_ids"].type(torch.long)
        item["attention_mask"] = item["attention_mask"].type(torch.long)
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs = False):
        labels = inputs.pop('labels')
        inputs["input_ids"] = inputs["input_ids"].squeeze(0)
        inputs["token_type_ids"] = inputs["token_type_ids"].squeeze(0)
        inputs["attention_mask"] = inputs["attention_mask"].squeeze(0)
        outputs = model(**inputs)
        logits = outputs
        loss = torch.nn.BCEWithLogitsLoss(reduction='mean')(logits,labels)
        return (loss, (logits, labels)) if return_outputs else loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        

        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
          with self.autocast_smart_context_manager():
            (loss, (outputs,labels)) =  self.compute_loss(model = model,inputs = inputs,return_outputs = True)
          #print("LOSS PREDICTION STEP:", loss)
          loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, outputs, labels)

class CLQTrainer(Trainer):
    def compute_loss(
        self, 
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False
        ):

        labels = inputs['labels']

        model.add_classes_to_memory(labels)
        prevalences = model.read_quantification_memory()

        inputs["input_ids"] = inputs["input_ids"].squeeze(0)
        inputs["token_type_ids"] = inputs["token_type_ids"].squeeze(0)
        inputs["attention_mask"] = inputs["attention_mask"].squeeze(0)

        classification_outputs,quantification_outputs = model(**inputs)

        #Classification Loss
        classification_logits = classification_outputs

        classification_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')(classification_logits,labels)

        quantification_loss = torch.nn.HuberLoss(reduction='mean',delta=HUBER_DELTA)(quantification_outputs,prevalences)
        quantification_loss = QUANT_LAMBDA *quantification_loss
        loss = sum([classification_loss,quantification_loss])

        return (loss, (classification_outputs,labels,quantification_outputs,prevalences)) if return_outputs else loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        

        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
          with self.autocast_smart_context_manager():
            (loss, (classification_outputs,classification_labels,quantification_outputs,quantification_prevalences)) =  self.compute_loss(model = model,inputs = inputs,return_outputs = True)
          #print("LOSS PREDICTION STEP:", loss)
          loss = loss.mean().detach()

          classification_outputs_logits = classification_outputs

          outputs = torch.cat((classification_outputs_logits,quantification_outputs),dim = 1)
          labels = torch.cat((classification_labels,quantification_prevalences),dim = 1)

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, outputs, labels)

class StatefulCallback(TrainerCallback):
    def __init__(self, model):
        self.quantification_group_min = model.quantification_group_min
        self.quantification_group_max = model.quantification_group_max
        self.model = model
        self.eval_counter = 0
    def on_epoch_begin(self, args, state, control, logs=None, **kwargs):
        self.model.set_quantification_group_size(min_value= self.quantification_group_min, max_value= self.quantification_group_max)
        self.model.reset_quantification_memory()
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        self.model.set_quantification_group_size(min_value= self.quantification_group_min, max_value= self.quantification_group_max)
        self.model.reset_quantification_memory()
        self.eval_counter = 0
    def on_step_end(self, args, state, control,model, logs=None, **kwargs):
        self.model.zero_grad()
        #print("READ ACUMULATIVE ITEMS EPOCH STEP: ", self.model.read_accumulative_items())
        quantify_group_size,batch_counter = self.model.read_accumulative_items()
        if batch_counter[-1] >= quantify_group_size:
            self.model.set_quantification_group_size(min_value=self.quantification_group_min,max_value=self.quantification_group_max)
            self.model.reset_quantification_memory()
    def on_substep_end(self, args, state, control,model, logs=None, **kwargs):
        #print("READ ACUMULATIVE ITEMS PREDICTION STEP: ", self.model.read_accumulative_items())
        quantify_group_size,batch_counter = self.model.read_accumulative_items()
        if batch_counter[-1] >= quantify_group_size:
            self.model.set_quantification_group_size(min_value=self.quantification_group_min,max_value=self.quantification_group_max)
            self.model.reset_quantification_memory()


def precision_at_k(y_true,y_pred, k: int=3) -> float:
    top_k_args = np.argsort(np.sum(y_true,axis = 0))[-k:]
    sliced_y_true = np.squeeze(y_true[:,[top_k_args]],axis = 1)
    sliced_y_pred = np.squeeze(y_pred[:,[top_k_args]],axis = 1)
    return precision_score(sliced_y_true,sliced_y_pred, average="micro")

def recall_at_k(y_true,y_pred, k: int=3) -> float:
    top_k_args = np.argsort(np.sum(y_true,axis = 0))[-k:]
    sliced_y_true = np.squeeze(y_true[:,[top_k_args]], axis = 1)
    sliced_y_pred = np.squeeze(y_pred[:,[top_k_args]], axis = 1)
    return recall_score(sliced_y_true,sliced_y_pred, average="micro")

def compute_metrics(pred):
    if os.path.isfile(SAVE_METRICS_PATH + '/metrics_result.pkl'):
        with open(SAVE_METRICS_PATH + '/metrics_result.pkl', 'rb') as f:
            metrics_list = pickle.load(f)
    else:
        metrics_list = []
    labels = pred.label_ids
    logits = pred.predictions
    if "CLQ" in MODEL_TYPE:
        classification_labels, quantification_prevalences = np.split(labels,2,axis = 1)
        classification_logits, quantification_outputs = np.split(logits,2,axis = 1)
    else:
        classification_labels = labels
        classification_logits = logits

    classification_preds = np.round(sigmoid(classification_logits))

    #Loss
    classification_list = []
    for logit, label in zip(classification_logits, classification_labels):
      class_loss = torch.nn.BCEWithLogitsLoss(reduction = "mean")(torch.Tensor(logit),torch.Tensor(label))
      classification_list.append(class_loss.item())
    classification_loss = np.mean(classification_list)

    if "CLQ" in MODEL_TYPE:
        quantification_list = []
        for logit, label in zip(quantification_outputs, quantification_prevalences):
            quant_loss = torch.nn.MSELoss()(torch.Tensor(logit),torch.Tensor(label))
            quantification_list.append(quant_loss.item())
        quantification_loss = np.mean(quantification_list)

        MSE_average = mean_squared_error(quantification_outputs,quantification_prevalences)
        MAE_average = mean_absolute_error(quantification_outputs,quantification_prevalences)

    classification_proba = torch.tensor(sigmoid(classification_logits))
    cls_labels = torch.tensor(classification_labels).type(torch.LongTensor)

    calibration_list = []
    for i in range(50):
        calibration_score = mece(classification_proba[:,i], cls_labels[:,i]).item()
        calibration_list.append(calibration_score)
    
    mece_score = np.mean(calibration_list)

    # Metrics
    f1_macro = f1_score(classification_labels, classification_preds, average='macro', zero_division=0)
    f1_micro = f1_score(classification_labels, classification_preds, average='micro', zero_division=0)

    precision_at_5 = precision_at_k(y_true=classification_labels,y_pred=classification_preds,k=5)
    precision_at_8 = precision_at_k(y_true=classification_labels,y_pred=classification_preds,k=8)
    precision_at_15 = precision_at_k(y_true=classification_labels,y_pred=classification_preds,k=15)

    recall_at_5 = recall_at_k(y_true=classification_labels,y_pred=classification_preds,k=5)
    recall_at_8 = recall_at_k(y_true=classification_labels,y_pred=classification_preds,k=8)
    recall_at_15 = recall_at_k(y_true=classification_labels,y_pred=classification_preds,k=15)

    

    metrics = 'classification: ' + str(classification_loss) + ', f1_macro: ' + str(f1_macro) +  ', f1_micro: ' + str(f1_micro) + ', precision_at_5: ' + str(precision_at_5) +  ', precision_at_8: ' + str(precision_at_8) + ', precision_at_15: ' + str(precision_at_15) + ', recall_at_5: ' + str(recall_at_5) +  ', recall_at_8: ' + str(recall_at_8) + ', recall_at_15: ' + str(recall_at_15) +', mece: '+ str(mece_score)
    
    if "CLQ" in MODEL_TYPE:
        metrics = metrics + ', quantification: ' + str(quantification_loss) + ', mse_avg: ' + str(MSE_average) + ', mae_avg: ' + str(MAE_average)
    
    metrics = metrics + '\n'
    metrics_list.append(metrics)

    with open(SAVE_METRICS_PATH + '/metrics_result.pkl', 'wb') as f:
        pickle.dump(metrics_list, f)
    
    if "CLQ" in MODEL_TYPE:
        return {
            'classification':classification_loss,
            'quantification':quantification_loss,
            'f1_macro': f1_macro, 
            'f1_micro': f1_micro,
            'precision_at_5': precision_at_5,
            'precision_at_8': precision_at_8,
            'precision_at_15': precision_at_15,
            'recall_at_5': recall_at_5,
            'recall_at_8': recall_at_8,
            'recall_at_15': recall_at_15,
            'mece': mece_score,
            "mse_avg":MSE_average,
            "mae_avg":MAE_average
        }
    else:
        return {
            'classification':classification_loss,
            'f1_macro': f1_macro, 
            'f1_micro': f1_micro,
            'precision_at_5': precision_at_5,
            'precision_at_8': precision_at_8,
            'precision_at_15': precision_at_15,
            'recall_at_5': recall_at_5,
            'recall_at_8': recall_at_8,
            'recall_at_15': recall_at_15,
            'mece': mece_score,
        }
    