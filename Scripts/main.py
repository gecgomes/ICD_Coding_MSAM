import numpy as np
import os
import pickle
from transformers import ( 
    AutoConfig, 
    AutoTokenizer,
    PretrainedConfig,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
import click
from config import * 
from condicional_config import *
from Helpers.main_helpers import *
from Helpers.model_helpers import ClassifierModel, CLQModel
from Helpers.trainer_helpers import LEDataset,CEDataset, MultiLabelTrainer, compute_metrics, CLQTrainer, StatefulCallback
from tqdm import tqdm
from Helpers.evaluation import all_metrics

def main():
    # Configure seed for reproducibility
    configure_seed(888)
    
    # ========= Create save files for metrics and checkpoint models =============

    if os.path.exists("../models/{}/{}".format(MODEL_TYPE,FILE_NAME)):
        print("WARNING: The directory named: <{}> already exists!".format("../models/{}/{}".format(MODEL_TYPE,FILE_NAME)))
        if not click.confirm('If you continue, you might overwrite some important files in this directory. Still want to continue?', default=True):
            print("EXIT")
            exit(1)

    if not os.path.exists("../models/{}".format(MODEL_TYPE)):
        os.makedirs("../models/{}".format(MODEL_TYPE))
    if not os.path.exists("../models/{}/{}".format(MODEL_TYPE,FILE_NAME)):
        os.makedirs("../models/{}/{}".format(MODEL_TYPE,FILE_NAME))
        os.makedirs(SAVE_METRICS_PATH)    
        os.makedirs(SAVE_PREDICTION_PATH)
        os.makedirs(OUTPUT_DIR)

    # ========== Load text and 1hot data =======
    print("Load MIMIC-III-{} dataset".format(DATA_TYPE))
    if DATA_TYPE == "top50":
        print("Load text...")
        with open(DATA_PATH + "/mimiciii_50_train.pkl", "rb") as file:
            train_data = pickle.load(file)
            train_texts = train_data["text"].tolist()
        with open(DATA_PATH + "/mimiciii_50_val.pkl", "rb") as file:
            val_data = pickle.load(file)
            val_texts = val_data["text"].tolist()
        with open(DATA_PATH + "/mimiciii_50_test.pkl", "rb") as file:
            test_data = pickle.load(file)
            test_texts = test_data["text"].tolist()

        print("Load 1hot labels...")
        train_1hot = np.load(DATA_PATH + '/train50_1hot.npz')['arr_0']
        val_1hot = np.load(DATA_PATH + '/val50_1hot.npz')['arr_0']
        test_1hot = np.load(DATA_PATH + '/test50_1hot.npz')['arr_0']

    elif DATA_TYPE == "clean":
        print("Load text...")
        with open(DATA_PATH + "/mimiciii_clean_train.pkl", "rb") as file:
            train_data = pickle.load(file)
            train_texts = train_data["text"].tolist()
        with open(DATA_PATH + "/mimiciii_clean_val.pkl", "rb") as file:
            val_data = pickle.load(file)
            val_texts = val_data["text"].tolist()
        with open(DATA_PATH + "/mimiciii_clean_test.pkl", "rb") as file:
            test_data = pickle.load(file)
            test_texts = test_data["text"].tolist()
            
        
        print("Load 1hot labels...")
        train_1hot = np.load(DATA_PATH + '/trainclean_1hot.npz')['arr_0']
        val_1hot = np.load(DATA_PATH + '/valclean_1hot.npz')['arr_0']
        test_1hot = np.load(DATA_PATH + '/testclean_1hot.npz')['arr_0']
    

    # ======== Load Models ========
    num_labels = len(train_1hot[0])
    quantification_group_min = 1
    quantification_group_max = len(train_texts)
    print("Load Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAIN_MODEL, do_lower_case=False)
    if "LE" in MODEL_TYPE:
        tokenizer.model_max_length = MAX_TEXT_LENGTH
        tokenizer.init_kwargs["model_max_length"] = MAX_TEXT_LENGTH
    
    print("Load Model...")
    if "CLQ" in MODEL_TYPE:
        config_clq = AutoConfig.from_pretrained(PRETRAIN_MODEL)
        config_clq.num_labels = num_labels
        config_clq.quantification_group_min = quantification_group_min
        config_clq.quantification_group_max = quantification_group_max
        config_clq.batch_size = 1

        if START_MODEL_FROM_CHECKPOINT != "":
            config_pretrained = PretrainedConfig.from_pretrained(START_MODEL_FROM_CHECKPOINT, num_labels=num_labels)
            config_pretrained.quantification_group_min = quantification_group_min
            config_pretrained.quantification_group_max = quantification_group_max
            config_pretrained.batch_size = 1
            model = CLQModel(config=config_clq).from_pretrained(START_MODEL_FROM_CHECKPOINT, config = config_pretrained)
        else:
            model = CLQModel(config=config_clq)
        model = model.to(DEVICE)
    else:
        config = AutoConfig.from_pretrained(PRETRAIN_MODEL, num_labels=num_labels, problem_type="multi_label_classification")
        config.model_name = PRETRAIN_MODEL
        config.attention_hidden_size = 1024
        config.M = M
        config.synonym_file =  "../data/{}_MSAM{}-{}_synonyms.pkl".format(DATA_TYPE,M,SELECTION_CRITERION)
        config.control_synonym = True
        model = ClassifierModel(config=config)
        if MODEL_FROM_CHECKPOINT != "":
            config_pretrained = PretrainedConfig.from_pretrained(PRETRAIN_MODEL, num_labels=num_labels)
            config_pretrained.model_name = config.model_name
            config_pretrained.M = config.M
            config_pretrained.synonym_file =  config.synonym_file
            config_pretrained.attention_hidden_size = config.attention_hidden_size
            config_pretrained.control_synonym = False
            model = model.from_pretrained(MODEL_FROM_CHECKPOINT, config = config_pretrained)

    # ======= Prepare Datasets =======
    if "LE" in MODEL_TYPE:
        train_dataset = LEDataset(train_texts, train_1hot,tokenizer = tokenizer)
        val_dataset = LEDataset(val_texts, val_1hot,tokenizer = tokenizer)
        test_dataset = LEDataset(test_texts, test_1hot,tokenizer = tokenizer)
    elif "CE" in MODEL_TYPE:
        train_dataset = CEDataset(train_texts, train_1hot,tokenizer = tokenizer)
        val_dataset = CEDataset(val_texts, val_1hot,tokenizer = tokenizer)
        test_dataset = CEDataset(test_texts, test_1hot,tokenizer = tokenizer)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')

    # ======= Prepare Training ======
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, 
        group_by_length=GROUP_BY_LENGTH,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        logging_strategy = LOGGING_STRATEGY,
        num_train_epochs=EPOCHS,                               
        per_device_train_batch_size=1,  
        per_device_eval_batch_size=1,                    
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        evaluation_strategy=EVALUATION_STRATEGY,
        save_strategy=SAVE_STRATEGY,
        dataloader_drop_last = True,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
        greater_is_better =GREATER_IS_BETTER ,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        optim=OPTIM
    )

    if "CLQ" in MODEL_TYPE:
        trainer = CLQTrainer(
            model=model,                         # the instantiated Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset,            # evaluation dataset
            compute_metrics=compute_metrics,     # the callback that computes metrics of interest
            data_collator=data_collator,         # to be able to build batches and add padding
            callbacks=[EarlyStoppingCallback(early_stopping_patience=SAVE_TOTAL_LIMIT - 1),],
        )

        trainer.add_callback(StatefulCallback(model = model))
    else:
        trainer = MultiLabelTrainer(
            model=model,                         # the instantiated Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset,            # evaluation dataset
            compute_metrics=compute_metrics,     # the callback that computes metrics of interest
            data_collator=data_collator,         # to be able to build batches and add padding
            callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
        )


    if MODE == "train":
        # ... Training model ... 
        print("======= LETS TRAIN ======")
        print("... Starting to Evaluate model (val set) ...")
        trainer.evaluate(eval_dataset = val_dataset)
        print("... Training model ...")
        trainer.train()
        print("... Evaluate model (test set) ...")
        trainer.evaluate(eval_dataset = test_dataset)
    else:
        print("======= LETS TEST ======")
        for dataset,texts,y_gt in zip(["val","test"],[val_dataset,test_dataset],[val_1hot,test_1hot]):
            print("Predict {} set ...".format(dataset))
            probabilities = np.zeros((len(texts),num_labels))

            y_pred = np.zeros((len(texts),num_labels))

            with tqdm(total=len(texts)) as pbar:
                for z, item in enumerate(texts):
                    model.eval()
                    with torch.no_grad():
                        item = {k: v.to(DEVICE) for k, v in item.items()}
                        if "CLQ" in MODEL_TYPE:
                            logits,_ = model(**item)
                            logits = logits[0].cpu().detach().numpy()
                        else:  
                            logits = model(**item)[0].cpu().detach().numpy()
                        
                    probabilities[z] = sigmoid(logits)
                    
                    y_pred[z] = np.round(sigmoid(logits))

                    pbar.update(1)

            metrics = all_metrics(y_pred, y_gt, k=[5,8,15], yhat_raw=probabilities)
            np.save(SAVE_PREDICTION_PATH + '/y_{}_prob.npy'.format(dataset), probabilities)

            np.save(SAVE_PREDICTION_PATH + '/y_{}_pred.npy'.format(dataset), y_pred)

            np.save(SAVE_PREDICTION_PATH + '/y_{}_true.npy'.format(dataset), y_gt)

            with open(SAVE_PREDICTION_PATH + "/{}-metrics.txt".format(dataset), "w") as file:
                file.write(str(metrics))
            
        if "CLQ" in MODEL_TYPE:
            #Save the MLP
            torch.save(model.mlp.state_dict(), OUTPUT_DIR + "/clq-mlp.pth")

if __name__ == '__main__':
    main()
