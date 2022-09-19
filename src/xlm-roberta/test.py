import utils
import dataset
import engine
import torch
import transformers
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from settings import get_module_logger
from model import RobertaBengali, RobertaBengaliTwo, RobertaBengaliNext, CustomRobertaBengali
from sklearn import model_selection
from transformers import AdamW
from dataset import BengaliDataset
from transformers import get_linear_schedule_with_warmup
from collections import defaultdict
from prediction import get_predictions
import gc
from flag import get_parser
parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
logger = get_module_logger(__name__)


path = "/content/drive/MyDrive/Bengali-Hate-Speech-Detection/xlm-roberta/store_model/"
test = pd.read_csv(args.testing_file).dropna().reset_index(drop = True)
test_dataset = BengaliDataset(
        text=test.text.values,
        targets=test.target.values
    )

test_data_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.valid_batch_size,
    shuffle=False,
    num_workers=2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model0 = RobertaBengali()
model0.to(device)
#model1 = BERTBengali()
#model1.to(device)
#model2 = BERTBengali()
#model2.to(device)


model0.load_state_dict(torch.load(f"{path}xlm-roberta-large-lr-3e-5.bin")) 
# model1.load_state_dict(torch.load(f"{path}bangla-bert-pool-base-lr-3e-5.bin"))
# model2.load_state_dict(torch.load(f"{path}bangla-bert-pool-base-lr-3e-5.bin"))

fin_outputs = []
with torch.no_grad():
    tk0 = tqdm(test_data_loader, total=len(test_data_loader))
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        mask = d["mask"]
        targets = d["targets"]
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)

        outputs = model0(
                    ids=ids,
                    attention_mask=mask,
                )
        # output1 = model1(
        #             ids=ids,
        #             attention_mask=mask,
        #             token_type_ids=token_type_ids
        #         )
        # output2 = model2(
        #             ids=ids,
        #             attention_mask=mask,
        #             token_type_ids=token_type_ids
        #         )
        #outputs = (output0 + output1 + output2) / 3
        outputs = torch.log_softmax(outputs,dim=1)
        outputs = torch.argmax(outputs,dim=1)
        fin_outputs.extend(outputs.cpu().detach().numpy().tolist())


test['pred'] = fin_outputs
print(test.head())
get_predictions(test) 
