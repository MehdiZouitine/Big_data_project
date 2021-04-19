import mlflow
from math import inf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def track(hyperparameters, metrics, model, epoch, save, run_id, path_cufusion_mat):
    """[Wrapper to log metrics,score and other informations on the mflow logger]

    Args:
        hyperparameters,
        metrics,
        model,
        epoch,
        save,
        run_id,
        path_cufusion_mat,
    """
    if epoch == 0:
        mlflow.log_params(hyperparameters)

        path = "model_layers/" + str(run_id) + ".txt"
        with open(path, "w") as text_file:
            text_file.write(str(model))
        mlflow.log_artifact(path)

    if save:
        mlflow.pytorch.log_model(model, "model_checkpoint")
        print("saving ...")

    mlflow.log_metrics(metrics, step=epoch)
    mlflow.log_artifact(path_cufusion_mat)


class EarlyStopping:
    def __init__(self, tolerance: int, delta: float):
        """[Early stopping module]

        Args:
            tolerance (int): [Number of epoch without improvement]
            delta (float): [Minimum difference of metric]
        """
        self.tolerance = tolerance  # MAX NUMBER OF EPOCH WITHOUT IMPROVEMENT
        self.nb_epoch_without_progress = 0  # NUMBER OF EPOCH WITHOUT IMPROVEMENT
        self.delta = delta  # MINIMUM VALUE OF IMPROVEMENT
        self.last_metric = -inf

    def __call__(self, val_metric) -> bool:
        if val_metric > self.last_metric + self.delta:
            self.nb_epoch_without_progress = 0
            self.last_metric = val_metric
            return {"stop": False, "save": True}
        else:
            self.nb_epoch_without_progress += 1
            if self.nb_epoch_without_progress >= self.tolerance:
                return {"stop": True, "save": False}
            return {"stop": False, "save": False}


def plot_confusion_matrix(array: np.ndarray, run_id: str, epoch: int):
    """[Plot confusion matric]

    Args:
        array (np.ndarray): [Confusion matrix]
        run_id (str): [mlflow run id]
        epoch (int): [number of epochs]

    Returns:
        [str]: [confusion matrix path]
    """
    array = array.astype("float") / array.sum(axis=1)[:, np.newaxis]
    array = np.round(array, 2)
    df_cm = pd.DataFrame(
        array, index=[i for i in range(28)], columns=[i for i in range(28)]
    )
    plt.figure(figsize=(20, 20))
    sns.heatmap(df_cm, annot=True)
    path = f"plot/CM_epoch{epoch}___{run_id}_.pdf"
    plt.savefig(path)
    return path
