import torch
import torch.nn as nn
import numpy as np
from flag import get_parser
parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def loss_fn(y_pred, y_true):
    return nn.BCEWithLogitsLoss()(y_pred, y_true.view(-1,1))

def get_prediction_value(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in enumerate(data_loader):
            ids = d["ids"]
            mask = d["mask"]
            targets = d["targets"]
            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(
                ids=ids,
                attention_mask=mask
            )
            outputs = torch.round(nn.Sigmoid()(outputs)).squeeze()
            targets = targets.squeeze()
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets