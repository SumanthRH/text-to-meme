# imports
import math, statistics, time
from collections import defaultdict
import numpy as np
from tqdm.autonotebook import tqdm as tqdm
from datetime import datetime
import pickle
import pandas as pd
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample


# HF token
token = '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'
from huggingface_hub import HfApi, HfFolder
api=HfApi()
folder=HfFolder()
api.set_access_token(token)
folder.save_token(token)
base_model = 'roberta-base'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_epochs = 10
model_save_path = './models/sentence_transformer_roberta_30'

with open('./data/training_label.pkl', 'rb') as f:
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

print(len(df_train),len(df_test))

class Dataset():
    def __init__(self, df):
        self.labels = [labels[label] for label in df['category']]
        self.texts = [text for text in df['text']]
    
    def __len__(self):
        return len(self.labels)

    def classes(self):
        return self.labels

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

train_dataset = Dataset(df_train)
test_dataset = Dataset(df_test)

class SentenceBertDataloader():
    def __init__(self, dataset, batch_size):
        self.batch_size=batch_size
        self.labels = np.array(dataset.labels)
        self.texts = np.array(dataset.texts)
        self.num_data_points = len(self.labels)
        self.num_meme_keys = len(set(self.labels))
        self.datapoints_per_meme = self.num_data_points//self.num_meme_keys
        
        # create mapping from meme id to list of texts for sampling +ve/-ve examples
        self.meme_id_text_dic = defaultdict(list)
        for meme_id, text in tqdm(zip(self.labels, self.texts)):
            self.meme_id_text_dic[meme_id].append(text)
        
        self.index = 0
    
    def __len__(self):
        return int(len(self.labels)//self.batch_size)
    
    def samplePositives(self, true_label, true_text):
        count = 0
        positive_examples = []
        while count<2:
            random_text = np.random.choice(self.meme_id_text_dic[true_label])
            if random_text!=true_text:
                count+=1
                positive_examples.append(random_text)
        return positive_examples
    
    def sampleNegatives(self, true_label, true_text):
        count = 0
        negative_examples = []
        while count<2:
            random_meme_id = np.random.randint(0, self.num_meme_keys)
            random_text = np.random.choice(self.meme_id_text_dic[random_meme_id])
            if random_meme_id!=true_label and random_text!=true_text:
                count+=1
                negative_examples.append(random_text)
        return negative_examples
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index==len(self):
            self.index=0
            
        start = self.index*self.batch_size
        end = self.index*self.batch_size + self.batch_size
        
        X = self.texts[start: end]
        y = self.labels[start: end]
        X_final_batch = []
        for i in range(0, len(X)):
            positive_examples = self.samplePositives(y[i], X[i])
            negative_examples = self.sampleNegatives(y[i], X[i])
            for example in positive_examples:
                X_final_batch.append(InputExample(texts=[X[i], example], label=1))
            for example in negative_examples:
                X_final_batch.append(InputExample(texts=[X[i], example], label=0))
        
        self.index+=1
        return self.collate_fn(X_final_batch)
    
train_loader = SentenceBertDataloader(train_dataset, 64)
test_loader = SentenceBertDataloader(test_dataset, 64)

model = SentenceTransformer('./models/sentence_transformer_roberta_20', device=device)
train_loss = losses.ContrastiveLoss(model=model)

model.fit(train_objectives=[(train_loader, train_loss)],epochs=num_epochs, output_path=model_save_path)