import pickle
import torch
import torch.nn as nn
import logging as log
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample
from sentence_transformers.util import cos_sim

base_model_name = "roberta_base"
models_path = '../models/{}'
embeddings_path = '../models/model_utils/{}/category_embeddings.pkl'

class Classifier:
    def __init__(self, num_classes=75, model_name=None, k=3):
        self.model_name = model_name if model_name else base_model_name
        self.model_path = models_path.format(self.model_name)
        self.embedding_path = embeddings_path.format(self.model_name)
        self.k=k
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # initialize clf
        self.model = SentenceTransformer(self.model_path)
    
        # load model embeddings
        with open(self.embedding_path, 'rb') as f:
            self.embeddings = pickle.load(f)

        # load training labels and create reverse dict
        with open('../data/training_label.pkl', 'rb') as f:
            self.uuid_labels_dict = pickle.load(f)
            self.labels_uuid_dict = {v: k for k, v in self.uuid_labels_dict.items()}
            
    def predictTopK(self, text):
        embedding = self.model.encode(text)        
        scores = [(cos_sim(embedding, v), key) for key, v in self.embeddings.items()]
        scores.sort(reverse=True)
        return [x[1] for x in scores[:self.k]]