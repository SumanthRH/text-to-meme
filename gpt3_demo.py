import flair
import transformers
from flair.data import Sentence
from flair.models import SequenceTagger
import openai
import pickle
from gpt3_sandbox.api import *
import numpy as np



def print_entities(tagger, sentence, span_str='np'):
    sentence = Sentence(sentence)
    tagger.predict(sentence)
    for entity in sentence.get_spans(span_str):
        print(entity)
        
def transform_caption_to_phrases(caption):
    sentence = Sentence(caption)
    tagger.predict(sentence)
    entities = []
    for entity in sentence.get_spans('np'):
#         import pdb; pdb.set_trace()
        if entity.get_label('np').value in ['NP', 'VP']:
            entities.append(entity.text)
    return ';'.join(entities)


def clean_and_unify_caption(caption):
    return caption[0].strip()+', '+caption[1].strip()

def read_data(file_path):
    # load data
    with open(path, 'rb') as f:
        gpt3_data = pickle.load(path)
    return gpt3_data


class PrimeGPT(object):
    def __init__(self, api_key, gpt3_data_path, gpt3_engine, temperature, max_tokens):
        set_openai_key(api_key)
        self.gpt = GPT(engine=gpt3_engine, temperature=temperature, max_tokens=max_tokens)
        self.gpt3_data = read_data(gpt3_data_path)
        
    def clear_gpt_examples(self):
        self.gpt.examples = {}
    
    def prime_gpt_from_uuid(self, uuid):
        self.clear_gpt_examples()
        transformed_captions = self.pt3_data['uuid_caption_dic'][uuid]
        captions = self.gpt3_data['uuid_trans_caption_dic'][uuid]
        for (transformed_caption, caption) in zip(transformed_captions, captions):
            self.gpt.add_example(Example(transformed_caption, caption))
    
    def get_response(self, uuid, caption):
        self.prime_gpt_from_uuid(uuid)
        prompt = transform_caption_to_phrases(caption)
        return self.gpt.submit_request(prompt)