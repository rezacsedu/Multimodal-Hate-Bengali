import pandas as pd 
import torch
import numpy as np
from ast import literal_eval
df = pd.read_csv("/content/drive/MyDrive/Bengali-Hate-Speech-Detection/bangla-bert-base/tesor.csv")
# print(type(torch.as_tensor(df.iloc[0]['tensor'])))
#print(literal_eval(df.iloc[0]['tensor']))
# tensor_value = torch.log_softmax(df.iloc[0]['tensor'])
# print(torch.argmax(tensor_value))
device = "cuda"
ll = [-6.3152e+00, -7.6447e+00, -6.2451e+00, -4.5511e-03, -8.0672e+00]
ll = np.array(ll)
print(ll)
print(ll.flatten())

