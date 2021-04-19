import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    BertModel,
    BertConfig,
    BertTokenizerFast,
    BertForSequenceClassification,
    AdamW,
    DistilBertModel,
    DistilBertTokenizerFast,
)
import mlflow
from dataloader import (
    NlpTrainDataset,
    NlpTestDataset,
)
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.optim import Adam
from focal import FocalLoss, LinkedFocalLoss
from train import learn
from logger import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from crossentropy import (
    HardNegCrossEntropy,
    LinkedCrossEntropy,
    LinkedHardNegCrossEntropy,
)
from model import BERT_clf
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    DATA_PATH = "src\deep\data"  # Specify the data path
    train_df = pd.read_json(DATA_PATH + "/train.json")  # data loading
    test_df = pd.read_json(DATA_PATH + "/test.json")
    train_label = pd.read_csv(DATA_PATH + "/train_label.csv")
    train_df["gender"] = train_df.apply(
        lambda x: 1 if x["gender"] == "M" else 0, axis=1
    )
    test_df["gender"] = test_df.apply(lambda x: 1 if x["gender"] == "M" else 0, axis=1)

    train_df["description"] = train_df["description"].apply(lambda x: x.lower())
    test_df["description"] = test_df["description"].apply(lambda x: x.lower())
    train, val, train_labels, val_labels = train_test_split(
        train_df,
        train_label["Category"],
        test_size=0.2,
        stratify=train_label["Category"],
        random_state=7,
        shuffle=True,
    )

    # Load bert tokenizer and bert model
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased")
    train_dataset = NlpTrainDataset(train, train_labels, tokenizer)
    val_dataset = NlpTrainDataset(val, val_labels, tokenizer)
    # Choose hyperparameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 1
    batch_size = 2
    lr = 1.5e-5
    lr_clf = None
    eps = 1e-8
    # Custom loss parameters
    alpha = None
    gamma = None
    gamma_scheduler = None
    freeze = False
    # Link use in custom loss (see loss file for more details)
    link = {0: [3, 19], 3: [19], 7: [24], 17: [11], 23: [26], 2: [3], 24: [13]}
    alpha_link = 3
    # For bert fine-tuning it's more suitable to use Adam-warmup
    optimizer_name = "AdamW"
    top_k = None
    loss_name = "Linked"
    scheduler_name = "get_linear_schedule_with_warmup"
    model = BERT_clf(
        bert, freeze=freeze
    )  # Load pretrained bert and choose to freeze or not the pretrained weight
    model = model.to(device)
    no_decay = [
        "bias",
        "LayerNorm.weight",
    ]  # We do not apply weight decay regularization on some layers (see bibliography)
    lr_layerdecay = None

    # Here is tricky part : We apply different learning rate and weight decray for each layer
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    ## SOME PARAMETERS TESTED
    # optimizer = Adam(model.parameters(),lr=lr,eps=eps)
    # loss_function = FocalLoss(alpha=alpha,gamma=gamma)
    # class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels.values)
    # weights= torch.tensor(class_weights,dtype=torch.float)
    # weights = weights.to(device)
    # loss_function  = nn.CrossEntropyLoss()
    # loss_function = HardNegCrossEntropy(top_k=top_k)
    loss_function = LinkedCrossEntropy(alpha_link, link)
    # loss_function = LinkedHardNegCrossEntropy(alpha_link, link,top_k)
    # CREATE LOADER TO PASS BATCH OF DATA INTO THE MODEL
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=3
    )  # Shuffle on train
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=3
    )
    total_steps = len(train_dataloader) * epochs
    num_warmup_steps = int(0.1 * total_steps)
    # Linearly decrease the lr and do some warmup step
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )
    # Early stopping hyperparameters
    tolerance = 20
    delta = 0.0005
    # Early stopping
    early_stopper = EarlyStopping(tolerance, delta)

    comment = "layer_decay"
    # All the parameters of the models (dict is usefull to log the model on mlflow)
    hyperparameters = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "lr_clf": lr_clf,
        "eps": eps,
        "alpha": alpha,
        "gamma": gamma,
        "optimizer": optimizer_name,
        "loss": loss_name,
        "scheduler": scheduler_name,
        "tolerance_es": tolerance,
        "delta_es": delta,
        "gamma_scheduler": gamma_scheduler,
        "top_k": top_k,
        "num_warmup": num_warmup_steps,
        "lr_layer_decay": lr_layerdecay,
        "freeze": freeze,
        "alpha_link": alpha_link,
        "comment": comment,
    }
    # Module used in the model
    modules = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "loss_function": loss_function,
        "device": device,
        "early_stopper": early_stopper,
    }
    # Train the model
    learn(
        train_dataloader,
        val_dataloader,
        modules,
        hyperparameters,
        experiment_name="seed",
    )
