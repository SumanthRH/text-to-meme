import numpy as np

class Dataset():
    def __init__(self, df, labels):
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