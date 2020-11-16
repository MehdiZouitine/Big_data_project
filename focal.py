import torch.nn as nn
import torch


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, y_pred, y_true):
        ce_loss = nn.functional.cross_entropy(y_pred, y_true, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss


class LinkedFocalLoss(nn.Module):
    def __init__(self, alpha_link: float, gamma: float, link: dict):
        super(LinkedFocalLoss, self).__init__()
        self.alpha_link = alpha_link
        self.gamma = gamma
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
        ce_loss = nn.functional.cross_entropy(y_pred, y_true, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = (link_bad * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss


# if __name__ == "__main__":
#     # pred = (
#     #     torch.tensor([[2, 7, 1], [4, 2, 1], [0, 2, 1], [1, 1, 2]], dtype=torch.float32)
#     #     .view(4, 3)
#     #     .cuda()
#     # )

#     # true = torch.tensor([0, 2, 1, 2], dtype=torch.long).cuda()
#     # link = {0: [1], 1: [2]}
#     # loss = LinkedFocalLoss(alpha_link=5, gamma=3, link=link)
#     # loss(pred, true)
