import pickle
import torch
import torch.nn as nn
import logging as log

from transformers import AutoTokenizer, AutoModel

pre_trained_model_checkpoint = "roberta-base"

class Meme_Classifier(nn.Module):
    def __init__(self, num_labels=75, dropout=0.3, device='cpu'):
        super(Meme_Classifier, self).__init__()
        self.model = AutoModel.from_pretrained(pre_trained_model_checkpoint)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, 512)
        self.linear2 = nn.Linear(512, num_labels)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.model(input_ids=input_id, attention_mask=mask,return_dict=False)
        dropout_output1 = self.dropout(pooled_output)
        linear_output1 = self.dropout(self.relu(self.linear1(dropout_output1)))
        final_output = self.relu(self.linear2(linear_output1))
        return final_output

class Classifier:
    def __init__(self, num_classes=75, model_name=None, k=3):
        self.default_classifier = "roberta-base-memes-900k-subset-75"
        self.model_name = model_name
        self.k=k
        self.model = Meme_Classifier(num_classes)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_checkpoint)

        # load training labels and create reverse dict
        fname = "training_label_100_cleaned" if num_classes == 100 else "training_label"
        with open(f'../data/{fname}.pkl', 'rb') as f:
            self.uuid_labels_dict = pickle.load(f)
            self.labels_uuid_dict = {v: k for k, v in self.uuid_labels_dict.items()}
            
        # check inputs
        if not self.model_name:
            self.model_name = self.default_classifier
            print("Model not passed. Using default model: ", self.model_name)
        self.MODEL_PATH = '../models/'+self.model_name
        
        # load state_dict
        self.model.load_state_dict(torch.load(self.MODEL_PATH, map_location=torch.device(self.device)))
        self.model.eval()

    def tokenizeAndFormat(self, text):
        tokenized = self.tokenizer(text,
                                   padding='max_length',
                                   max_length=50,
                                   truncation=True,
                                   return_tensors="pt")

        return tokenized['input_ids'], tokenized['attention_mask']

    def findTopK(self, logits):
        return [self.labels_uuid_dict[meme_id.item()] for meme_id in torch.topk(logits[0], self.k).indices]

    def predictTopK(self, text=None):

        if not text:
            print("no text passed by user..exiting")
            exit()

        # predict
        input_ids, mask = self.tokenizeAndFormat(text)
        with torch.no_grad():
            logits = self.model(input_ids, mask)

        # return top k entries
        return self.findTopK(logits)