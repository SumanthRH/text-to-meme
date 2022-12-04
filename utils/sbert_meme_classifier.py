import pickle
import torch
import torch.nn as nn
import logging as log
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample
from sentence_transformers.util import cos_sim
from collections import defaultdict
from tqdm import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'

base_model_name = "roberta_base"
models_path = 'models/{}'
embeddings_path = 'models/model_utils/{}/category_embeddings.pkl'

class Classifier:
    def __init__(self, model_name=None, k=3):
        self.model_name = model_name if model_name else base_model_name
        self.model_path = models_path.format(self.model_name)
        self.embedding_path = embeddings_path.format(self.model_name)
        self.k = k
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # initialize clf
        self.model = SentenceTransformer(self.model_path)
    
        # load model embeddings
        with open(self.embedding_path, 'rb') as f:
            self.embeddings = pickle.load(f)
    def predictTopK(self, text):
        embedding = self.model.encode(text)
        scores = [(cos_sim(embedding, v), key) for key, v in self.embeddings.items()]
        scores.sort(reverse=True)
        return [x[1] for x in scores[:self.k]]
    

    def _topKPrediction(self, ks, sentences, true_uuids):
        embeddings = self.model.encode(sentences)
        final_scores_dict = defaultdict(int)
        for i in range(len(sentences)):        
            scores = [(cos_sim(embeddings[i], v), key) for key, v in self.embeddings.items()]
            scores.sort(reverse=True)
            top_uuids = [x[1] for x in scores]
            for k in ks:
                if true_uuids[i] in top_uuids[:k]:
                    final_scores_dict[k]+=1
        return final_scores_dict
    
    def topKAccuracy(self, ks, df_test):
        accuracy = 0
        texts = list(df_test.text)
        true_meme_uuids = list(df_test.category)
        batch_size = 512
        final_scores = defaultdict(int)
        for i in tqdm(range(0,len(texts), batch_size)):
            scores_dict = self._topKPrediction(ks, 
                                         texts[i:i+batch_size], 
                                         true_meme_uuids[i:i+batch_size])
            final_scores = {key: final_scores[key]+scores_dict[key] for key in scores_dict}
        for k, v in final_scores.items():
            final_scores[k] = v/len(texts)
        return final_scores
