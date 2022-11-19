import numpy as np
import pandas as pd
from utils.dataset import Dataset
from collections import defaultdict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample


class SentenceBertDataloader():
    def __init__(self, dataset, batch_size):
        self.batch_size=batch_size
        self.labels = np.array(dataset.labels)
        self.texts = np.array(dataset.texts)
        self.num_data_points = len(self.labels)
        self.num_meme_keys = len(set(self.labels))
        self.meme_keys = list(set(self.labels))
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
            random_meme_id = np.random.choice(self.meme_keys)
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