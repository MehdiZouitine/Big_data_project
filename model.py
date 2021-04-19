import torch.nn as nn
import torch


class BERT_clf(nn.Module):
    @staticmethod
    def init_clf_head(module):
        module.weight.data.normal_(mean=0.0, std=0.02)
        module.bias.data.zero_()

    def __init__(self, bert, freeze):

        super(BERT_clf, self).__init__()

        self.bert = bert
        self.dropout = nn.Dropout(0.1)

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(768, 28)
        self.fc1.apply(self.init_clf_head)

        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, sent_id, mask):

        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)  # BERT

        # Bert fine tuning
        x = self.dropout(cls_hs)
        x = self.fc1(x)

        return x

    def unfreeze(self):
        for param in self.bert.parameters():
            param.requires_grad = True
