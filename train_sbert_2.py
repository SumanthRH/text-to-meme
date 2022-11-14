# imports
import sys
sys.path.append('./')
import math, statistics, time
from collections import defaultdict
import numpy as np
from datetime import datetime
import pickle
import pandas as pd
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample
from utils.dataset import Dataset
from utils.sentence_bert_dataloader import SentenceBertDataloader

base_model = 'roberta-base'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_epochs = 5
num_classes = 24
model_save_path = './models/sentence_transformer_num_classes_{}_num_epochs_{}'.format(num_classes, num_epochs)

with open('./data/training_label_{}.pkl'.format(num_classes), 'rb') as f:
    labels = pickle.load(f)
    
# load meme dataset
meme_dict = None
with open('./data/meme_900k_cleaned_data_v2.pkl', 'rb') as f:
    meme_dict = pickle.load(f)
print("Keys in meme dict dataset:", meme_dict.keys())
print("Number of uuids:", len(meme_dict['uuid_label_dic']))

# utility functions
def clean_and_unify_caption(caption):
    return caption[0].strip()+'; '+caption[1].strip()

# create pandas dataframe
training_uuids = labels.keys()
temp_arr = []
for uuid in training_uuids:
    for caption in meme_dict['uuid_caption_dic'][uuid]:
        temp_arr.append([uuid, clean_and_unify_caption(caption)])
df = pd.DataFrame(temp_arr, columns=['category', 'text'])

# split dataset
np.random.seed(42)
df_train, df_test = np.split(df.sample(frac=1, random_state=42), [int(.9*len(df))])
print(len(df_train), len(df_test))

# create datasets
train_dataset = Dataset(df_train, labels)
test_dataset = Dataset(df_test, labels)

# create dataloaders    
train_loader = SentenceBertDataloader(train_dataset, 64)
test_loader = SentenceBertDataloader(test_dataset, 64)

model = SentenceTransformer(base_model, device=device)
train_loss = losses.ContrastiveLoss(model=model)

model.fit(train_objectives=[(train_loader, train_loss)],epochs=num_epochs, output_path=model_save_path)