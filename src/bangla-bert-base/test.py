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
from model import BERTBengali, BERTBengaliTwo, BERTBengaliNext, CustomBERTBengali
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


path = "/content/drive/MyDrive/Bengali-Hate-Speech-Detection/bangla-bert-base/store_model/"
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


model0 = BERTBengali()
model0.to(device)
#model1 = BERTBengali()
#model1.to(device)
#model2 = BERTBengali()
#model2.to(device)


model0.load_state_dict(torch.load(f"{path}bangla-bert-pool-base-lr-3e-5.bin")) 
# model1.load_state_dict(torch.load(f"{path}bangla-bert-pool-base-lr-3e-5.bin"))
# model2.load_state_dict(torch.load(f"{path}bangla-bert-pool-base-lr-3e-5.bin"))

fin_outputs = []
tensor_save = []
with torch.no_grad():
    tk0 = tqdm(test_data_loader, total=len(test_data_loader))
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        mask = d["mask"]
        token_type_ids = d["token_type_ids"]
        targets = d["targets"]
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)

        outputs = model0(
                    ids=ids,
                    attention_mask=mask,
                    token_type_ids=token_type_ids
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
        print(outputs)
        outputs = torch.softmax(outputs,dim=1)
        for index in range(outputs.shape[0]):
            tensor_save.append(outputs[index])
        outputs = torch.argmax(outputs,dim=1)
        fin_outputs.extend(outputs.cpu().detach().numpy().tolist())


test['pred'] = fin_outputs
test['tensor'] = tensor_save
test.to_csv("/content/drive/MyDrive/Bengali-Hate-Speech-Detection/bangla-bert-base/tesor.csv",index=False)
get_predictions(test)




# m1 m2 m3  m 
# 1  1  0   1 
# 2  4  2   2
# 1  1  1   1
# 0  1  1   1 
# 1  2  3   e
