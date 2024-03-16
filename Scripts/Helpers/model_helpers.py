import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from opt_einsum import contract
from config import *
from condicional_config import *
import numpy as np
import os
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, AutoConfig,PretrainedConfig
from Helpers.bert2longformer import convert_bert_to_longformer
from Helpers.main_helpers import *

class SequenceClassifierOutput():
    """
    Classification output class
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class LongformerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, hidden_states, **kwargs):
        hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output

class MultiSynonymsAttention(nn.Module):
    def __init__(self, d_model,n_heads):
        super(MultiSynonymsAttention, self).__init__()

        self.d_k = int(d_model /n_heads)
        self.d_v = self.d_k
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, self.d_k * n_heads)
        self.W_K = nn.Linear(d_model, self.d_k * n_heads)
        self.W_V = nn.Linear(d_model, self.d_k * n_heads)

    def forward(self, Q, H, ql):
        n_classes = Q.size(0)
        batch_size = H.size(0)

        q_s = self.W_Q(Q).view(n_classes, -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: Synonym Embeddings [#Codes, #Synonyms, #Heads, #Hidden/Heads]
        k_s = self.W_K(H).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: Text Embeddings [#Chunks, #Tokens,  #Heads, #Hidden/Heads]
        Wql = self.W_V(ql)  # v_s: Global Code Embeddings [#Codes, #Heads, #Hidden/Heads] 
        H = H.view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
            
        scores,context, attn = SynonymsScaledDotProductAttention(self.d_k)(q_s, k_s, H)

        context = context.view(batch_size,n_classes,self.n_heads * self.d_v) 

        output = contract("bch,ch->bc",context,Wql)  

        return output # output: [batch_size x len_q x d_model]

class SynonymsScaledDotProductAttention(nn.Module):
    def __init__(self,d_k):
        super(SynonymsScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, WQ, WK,K):
        WK = nn.Tanh()(WK)
        K = nn.Tanh()(K)

        scores = contract('bzth,czsh->bczst',WK, WQ)
        scores = scores.to(DEVICE)
        attn = nn.Softmax(dim=-1)(scores)
        context = contract('bzth,bczst->bczsh',K,attn)
        context = nn.AvgPool3d((1,context.shape[3],1))(context).squeeze(-2)

        return scores, context, attn
    
class LE(PreTrainedModel):
    config_class = AutoConfig
    def __init__(self,config):
        super(LE, self).__init__(config)

        m_model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=config.num_labels).to(DEVICE)
        model = convert_bert_to_longformer(
                bert_model=m_model,
                longformer_max_length=8192,
            )
        self.model = model.to(DEVICE)
        self.clsHead = LongformerClassificationHead(config)
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None):
        
        encoder_outputs = self.model.bert(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    return_dict = False
            )
        sentences_encodings = encoder_outputs[0]
        predictions = self.clsHead(sentences_encodings)
        return predictions

class CE(PreTrainedModel):
    config_class = AutoConfig
    def __init__(self,config):
        super(CE, self).__init__(config)

        self.model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=config.num_labels).to(DEVICE)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None): 
        classification_outputs = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
        ).logits
        transposed_classification_outputs = classification_outputs.T
        max_pooling_outputs = nn.MaxPool1d(classification_outputs.shape[0], stride=classification_outputs.shape[0])(transposed_classification_outputs).T
        predictions = max_pooling_outputs
        return predictions

class MSAM(PreTrainedModel):
    config_class = AutoConfig
    def __init__(self,config):
        super(MSAM, self).__init__(config)

        model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=config.num_labels).to(DEVICE)
        if "LE" in MODEL_TYPE:
            model = convert_bert_to_longformer(
                bert_model=model,
                longformer_max_length=8192,
            )
        self.model = model.to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        if not os.path.exists(OUTPUT_DIR + "/synonms_matrix.pkl"):
            with open(config.synonym_file, 'rb') as f:
                code_synonms =  pickle.load(f)
            synonms = code_synonms
            self.label_tokens = []
            self.synonms_matrix = []
            self.model.eval()
            with torch.no_grad():
                for s in synonms:
                    s = list(s)
                    i =  self.tokenizer(s ,return_tensors='pt', padding='max_length',truncation = True, max_length=512)["input_ids"].to(DEVICE)
                    t =  self.tokenizer(s ,return_tensors='pt', padding='max_length',truncation = True, max_length=512)["token_type_ids"].to(DEVICE)
                    a =  self.tokenizer(s ,return_tensors='pt', padding='max_length',truncation = True, max_length=512)["attention_mask"].to(DEVICE)
                    out = self.model.bert(
                        input_ids=i,
                        token_type_ids=t,
                        attention_mask=a,
                        return_dict = False
                    )[0]
                    self.label_tokens.append(i)
                    out = self.Qlabel_pooling(out)
                    self.synonms_matrix.append(out.clone())
                    free_tensor(out)
                    torch.cuda.empty_cache()
            self.synonms_matrix = pad_sequence(self.synonms_matrix,batch_first = True,padding_value=0).to(DEVICE)
            self.synomns_vector = self.qllabel_pooling(self.synonms_matrix)
            print("SYNOMS MATRIX SHAPE",self.synonms_matrix.shape)
            print("SYNOMS VECTOR SHAPE",self.synomns_vector.shape)
            with open(OUTPUT_DIR + "/synonms_matrix.pkl", 'wb') as f:
                pickle.dump(self.synonms_matrix,f)
        else:
            with open(OUTPUT_DIR + "/synonms_matrix.pkl", 'rb') as f:
                self.synonms_matrix = pickle.load(f)
            self.synomns_vector = self.qllabel_pooling(self.synonms_matrix)
        self.attention = MultiSynonymsAttention(config.attention_hidden_size,config.M)

    def Qlabel_pooling(self,label_ids):
        return label_ids[:,0,:]
    
    def qllabel_pooling(self,synomn_matrix):
        return nn.AvgPool1d(synomn_matrix.shape[1],stride = synomn_matrix.shape[1])(synomn_matrix.permute(0,2,1)).squeeze(2)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None): 
        
        encoder_outputs = self.model.bert(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    return_dict = False
            )
        sentences_encodings = encoder_outputs[0]
        q =self.synonms_matrix # Synonyms
        ql = self.synomns_vector # Global representation of Codes
        h = sentences_encodings #Input
        classification_outputs = self.attention(q,h,ql)
        predictions = nn.MaxPool1d(classification_outputs.shape[0], stride=classification_outputs.shape[0])(classification_outputs.permute(1,0)).T
        return predictions

class ClassifierModel(PreTrainedModel):
    config_class = AutoConfig
    def __init__(self,config):
        super(ClassifierModel,self).__init__(config)

        if "MSAM" in MODEL_TYPE:
            self.model = MSAM(config)
        elif "LE" in MODEL_TYPE:
            self.model = LE(config)
        elif "CE" in MODEL_TYPE:
            self.model = CE(config)
        else:
            print("ERROR: Invalid MODEL_TYPE = {}".format(MODEL_TYPE))
            exit(1)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None):
        
        return self.model(input_ids,token_type_ids,attention_mask,labels)

class MLP(nn.Module):
    def __init__(self, hidden_size, n_classes):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.head_layer = nn.Linear(in_features = self.n_classes, out_features = self.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.final_layer = nn.Linear(in_features = self.hidden_size, out_features = self.n_classes)
    def forward(self, x):
        y_head = self.sigmoid(self.head_layer(x))
        y = self.sigmoid(self.final_layer(y_head))
        return y


class CLQModel(PreTrainedModel):
    config_class = AutoConfig
    def __init__(self,config):
        super(CLQModel,self).__init__(config)
        self.num_labels = config.num_labels
        self.quantification_group_min = config.quantification_group_min
        self.quantification_group_max = config.quantification_group_max
        self.batch_size = config.batch_size

        #Init true prevalence training memory
        self.quantification_group_size = 0
        self.classes_counter = torch.zeros((self.batch_size,self.num_labels)).to(DEVICE)
        self.batch_counter = torch.arange(-self.batch_size + 1,1).unsqueeze(1).type(torch.float32).to(DEVICE)

        self.predictions_counter = torch.zeros((self.batch_size,self.num_labels)).to(DEVICE)

        #Load MultiLabel Classification Model with a given checkpoint and extract its body
        config_classifier = AutoConfig.from_pretrained(PRETRAIN_MODEL, num_labels=self.num_labels, problem_type="multi_label_classification")
        config_classifier.model_name = PRETRAIN_MODEL
        config_classifier.attention_hidden_size = 1024
        config_classifier.M = M
        config_classifier.synonym_file =  "../data/{}_MSAM{}-{}_synonyms.pkl".format(DATA_TYPE,M,SELECTION_CRITERION)
        config_classifier.control_synonym = True
        model = ClassifierModel(config=config_classifier)
        if MODEL_FROM_CHECKPOINT != "":
            config_pretrained = PretrainedConfig.from_pretrained(PRETRAIN_MODEL, num_labels=self.num_labels)
            config_pretrained.model_name = config.model_name
            config_pretrained.M = config.M
            config_pretrained.synonym_file =  config.synonym_file
            config_pretrained.attention_hidden_size = config.attention_hidden_size
            config_pretrained.control_synonym = False
            model = model.from_pretrained(MODEL_FROM_CHECKPOINT, config = config_pretrained)
        
        self.classifier = model
        self.mlp = MLP(MLP_HIDDEN_SIZE, self.num_labels)
        if START_CLQ_FROM_QUANTIFIER_CHECKPOINT != "":
            self.mlp.load_state_dict(torch.load(START_CLQ_FROM_QUANTIFIER_CHECKPOINT))
        self.sigmoid = nn.Sigmoid()

    
    def reset_quantification_memory(self):
        self.classes_counter.zero_()
        self.batch_counter = torch.arange(-self.batch_size + 1,1).unsqueeze(1).type(torch.float32).to(DEVICE)
        self.predictions_counter.zero_()
    
    def set_quantification_group_size(self,min_value,max_value = None):
        random.seed(time.time())
        if max_value != None:
            self.quantification_group_size = random.randint(min_value,max_value)
        elif max_value == None:
            self.quantification_group_size = min_value

    def add_classes_to_memory(self,labels):
        self.classes_counter = ((torch.ones(self.classes_counter.shape).to(DEVICE))*self.classes_counter[-1,:]) + torch.cumsum(labels, dim=0)
        self.batch_counter += (torch.ones((self.batch_size,1))*self.batch_size).to(DEVICE)

    
    def read_quantification_memory(self): 
        if not torch.equal(self.batch_counter, torch.zeros((self.batch_size,1)).to(DEVICE)):
            return self.classes_counter/self.batch_counter
        else:
            return "Error: 0 read instances"
    
    def read_accumulative_items(self):
        return (self.quantification_group_size,self.batch_counter)

    
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None):
        
        classification_outputs = self.classifier(
            input_ids=input_ids,
            token_type_ids = token_type_ids,
            attention_mask=attention_mask,
            labels = labels
        )

        self.predictions_counter = self.predictions_counter.detach()
        
        self.predictions_counter += self.sigmoid(classification_outputs)

        prevalences = self.predictions_counter/self.batch_counter
        mlp_inputs = prevalences
        quantification_outputs_probabilities = self.mlp(mlp_inputs)

        return classification_outputs,quantification_outputs_probabilities

