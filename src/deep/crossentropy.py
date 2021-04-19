import torch.nn as nn
import torch


class HardNegCrossEntropy(nn.Module):
    def __init__(self, top_k: int, weight: torch.tensor = None):
        """[Custom cross entropy that multiply the gradient on top k error sample of the batch]

        Args:
            top_k (int): [Number of sample to penalize on the batch]
            weight (torch.tensor, optional): [Optional sample weight for cross entropy]. Defaults to None.
        """
        super(HardNegCrossEntropy, self).__init__()
        self.top_k = top_k
        self.weight = weight

    def __call__(self, y_pred, y_true):

        ce_loss = nn.functional.cross_entropy(
            y_pred, y_true, weight=self.weight, reduction="none"
        )
        worst_k, _ = torch.topk(input=ce_loss, k=self.top_k)

        loss = worst_k.mean()
        return loss


class LinkedCrossEntropy(nn.Module):
    def __init__(self, alpha_link: float, link: dict, weight: torch.tensor = None):
        """[Which penalizes the fact of misclassifying two related classes. We link two classes when they are close semantically:
         For example we will strongly penalize a model that predicts
          teacher instead of professor because they are related.
           We do this because the model tends to confuse two close classes.
            The model tends to predict teacher for teacher because the class teacher
             is in the majority.]

        Args:
            alpha_link (float): [Coefficient of penalization]
            link (dict): [Dict of link beetween class]
            weight (torch.tensor, optional): [Optional sample weight for cross entropy]. Defaults to None.
        """
        super(LinkedCrossEntropy, self).__init__()
        self.alpha_link = alpha_link
        self.weight = weight
        self.link = link

    def __call__(self, y_pred, y_true):
        device = y_pred.device
        link_bad = []
        for pred_label, gt in zip(torch.argmax(y_pred, dim=1), y_true):
            pred_label = pred_label.item()
            gt = gt.item()
            if pred_label == gt or (gt not in self.link):
                link_bad.append(1)
            else:
                if pred_label in self.link[gt]:
                    link_bad.append(self.alpha_link)
                else:
                    link_bad.append(1)

        link_bad = torch.tensor(link_bad, device=device)
        ce_loss = nn.functional.cross_entropy(
            y_pred, y_true, weight=self.weight, reduction="none"
        )
        loss = (link_bad * ce_loss).mean()
        return loss


class LinkedHardNegCrossEntropy(nn.Module):
    def __init__(self, alpha_link: float, link: dict, top_k:int, weight:torch.tensor=None):
        """[Fusion of the 2 precedent losses]

        Args:
            alpha_link (float): [Coefficient of penalization]
            link (dict): [Dict of link beetween class]
            top_k (int): [Number of sample to penalize on the batch]
            weight (torch.tensor, optional): [Optional sample weight for cross entropy]. Defaults to None.
        """
        
        super(LinkedHardNegCrossEntropy, self).__init__()
        self.alpha_link = alpha_link
        self.weight = weight
        self.link = link
        self.top_k = top_k

    def __call__(self, y_pred, y_true):
        device = y_pred.device
        link_bad = []
        for pred_label, gt in zip(torch.argmax(y_pred, dim=1), y_true):
            pred_label = pred_label.item()
            gt = gt.item()
            if pred_label == gt or (gt not in self.link):
                link_bad.append(1)
            else:
                if pred_label in self.link[gt]:
                    link_bad.append(self.alpha_link)
                else:
                    link_bad.append(1)

        link_bad = torch.tensor(link_bad, device=device)
        ce_loss = nn.functional.cross_entropy(
            y_pred, y_true, weight=self.weight, reduction="none"
        )
        ce_loss = link_bad * ce_loss
        worst_k, _ = torch.topk(input=ce_loss, k=self.top_k)

        loss = worst_k.mean()
        return loss


if __name__ == "__main__":
    pred = (
        torch.tensor(
            [
                [2, 7, 1],
                [4, 2, 1],
                [0, 2, 1],
                [1, 1, 2],
                [1, 1, 2],
                [1, 1, 2],
                [1, 1, 2],
            ],
            dtype=torch.float32,
        )
        .view(7, 3)
        .cuda()
    )
    link = {0: [1], 1: [2]}
    true = torch.tensor([0, 2, 1, 2, 0, 0, 0], dtype=torch.long).cuda()
    loss = LinkedHardNegCrossEntropy(alpha_link=5, link=link, top_k=5)
    print(loss(pred, true))
