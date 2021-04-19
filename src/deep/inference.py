import torch
import numpy as np
from tqdm import tqdm

def make_submission(model, model_path, dataloader):
    model = model.load_state_dict(torch.load(path))
    total_preds = []
    for batch in tqdm(dataloader):
        preds = model(batch["input_ids"], batch["attention_mask"])
        preds = preds.logits.detach().cpu().numpy()
        total_preds.append(preds)
        del preds
    result = np.argmax(np.concatenate(total_preds), axis=1)
