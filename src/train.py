import time
import mlflow
import numpy as np

from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn

from logger import track, plot_confusion_matrix
from sklearn.metrics import confusion_matrix


def train(train_dataloader, modules, hyperparameters):
    model = modules["model"]
    optimizer = modules["optimizer"]
    scheduler = modules["scheduler"]
    loss_function = modules["loss_function"]
    device = modules["device"]
    model.train()

    total_loss = 0

    # empty list to save model predictions
    total_preds = []
    total_labels = []

    # iterate over batches
    for step, batch in enumerate(train_dataloader):

        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print("  Batch {:>5,}  of  {:>5,}.".format(step, len(train_dataloader)))

        sent_id = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        gender = batch["gender"].to(device)

        model.zero_grad()

        preds = model(sent_id, mask)

        # preds = model(sent_id, attention_mask=mask).logits
        loss = loss_function(preds, labels)

        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()
        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()
        scheduler.step()

        # model predictions are stored on GPU. So, push it to CPU
        # preds = preds.detach().cpu().numpy()

        # labels = labels.detach().cpu().numpy().tolist()

        # append the model predictions
        # total_preds.append(preds)

        # total_labels += labels
    # compute the training loss of the epoch
    # avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    # total_preds = np.concatenate(total_preds, axis=0)

    # returns the loss and predictions
    # return avg_loss, total_preds, total_labels


# function for evaluating the model
def evaluate(dataloader, modules, hyperparameters):
    model = modules["model"]
    loss_function = modules["loss_function"]
    device = modules["device"]

    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss = 0

    # empty list to save the model predictions
    total_preds = []
    total_labels = []

    # iterate over batches
    for step, batch in enumerate(dataloader):

        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:

            # Report progress.
            print("  Batch {:>5,}  of  {:>5,}.".format(step, len(dataloader)))

        # push the batch to gpu
        sent_id = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        gender = batch["gender"].to(device)

        # deactivate autograd
        with torch.no_grad():

            # model predictions
            # preds = model(sent_id, attention_mask=mask).logits
            preds = model(sent_id, mask)

            loss = loss_function(preds, labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            labels = labels.detach().cpu().numpy().tolist()

            total_preds.append(preds)
            total_labels += labels

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(dataloader)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds, total_labels


def learn(
    train_dataloader, val_dataloader, modules, hyperparameters, experiment_name="bert"
):
    # set initial loss to infinite
    epochs = hyperparameters["epochs"]

    # empty lists to store training and validation loss of each epoch
    model = modules["model"]
    early_stopper = modules["early_stopper"]

    # for each epoch
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run() as run:
        for epoch in range(epochs):
            start = time.time()

            print("\n Epoch {:} / {:}".format(epoch + 1, epochs))

            # train model
            train(train_dataloader, modules, hyperparameters)

            train_loss, train_pred, train_lab = evaluate(
                train_dataloader, modules, hyperparameters
            )

            # evaluate model
            valid_loss, val_pred, val_lab = evaluate(
                val_dataloader, modules, hyperparameters
            )

            res_train = np.argmax(train_pred, axis=1).tolist()
            res_val = np.argmax(val_pred, axis=1).tolist()

            f1_train = f1_score(train_lab, res_train, average="macro")
            f1_val = f1_score(val_lab, res_val, average="macro")

            acc_train = accuracy_score(train_lab, res_train)
            acc_val = accuracy_score(val_lab, res_val)
            confusion = confusion_matrix(val_lab, res_val)
            path_confusion_mat = plot_confusion_matrix(
                confusion, run.info.run_id, epoch
            )
            # scheduler.step()

            metrics = {
                "train_loss": train_loss,
                "val_loss": valid_loss,
                "f1_train": f1_train,
                "f1_val": f1_val,
                "acc_train": acc_train,
                "acc_val": acc_val,
            }
            stop_and_save = early_stopper(f1_val)
            track(
                hyperparameters,
                metrics,
                model,
                epoch,
                stop_and_save["save"],
                run.info.run_id,
                path_confusion_mat,
            )
            print(f"\nTraining Loss: {train_loss:.3f}, Training f1 : {f1_train}")
            print(f"Validation Loss: {valid_loss:.3f}, Validation f1 : {f1_val}")

            now = time.time()
            print(f"Time for epoch {epoch} is {(now - start)/60} min")
            if stop_and_save["stop"]:
                break
        return run.info.run_id


# def make_submission(run_id):
#     path = f"mlruns/4/{run_id}/artifacts/model_checkpoint/data/model.pth"
#     model = torch.load(path)
