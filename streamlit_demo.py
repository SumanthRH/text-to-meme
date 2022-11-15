# imports
import sys
# sys.path.append('.')
from utils.sbert_meme_classifier import Classifier
import pickle
from PIL import Image
import os
import matplotlib.pyplot as plt
from gpt3_demo import PrimeGPT
import openai
from PIL import Image, ImageOps, ImageFont, ImageDraw
from utils.draw_utils import draw_caption_and_display
import streamlit as st

datapath = "data/gpt3_user_prompt_dic.pkl"
# TODO: Add a button for the API Key
your_personal_api_key = "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"

# constants
gpt3_engine = 'text-davinci-002'
temperature=0.8
max_tokens=256
frequency_penalty=0.0
presence_penalty=0.0
DATA_PATH = "data/"

def get_data_and_models():
    with open("data/meme_900k_cleaned_data_v2.pkl", 'rb') as f:
        data = pickle.load(f)
    model_name = 'sentence_transformer_roberta_20'
    clf = Classifier(model_name=model_name, k=15)
    gpt = PrimeGPT(your_personal_api_key, datapath, gpt3_engine, temperature, max_tokens)
    return data, clf, gpt
st.title("Jester")

if "data" not in st.session_state:
    data, clf, gpt = get_data_and_models()
    st.session_state['data'] = data
    st.session_state['clf'] = clf
    st.session_state['gpt'] = gpt

# =(1, 10, 1)
def show_images():
    ind = st.session_state.image_ind
    file_name = st.session_state.paths[ind - 1]
    img = Image.open(os.path.join(DATA_PATH, file_name))
    img = img.convert(mode="RGB")
    st.session_state.img = img

def get_templates():
    # initialize classifier
    predictions = st.session_state.clf.predictTopK(text=st.session_state.prompt)
    paths = [st.session_state.data['uuid_image_path_dic'][uuid] for uuid in predictions]
    #     display(paths)
    st.session_state.paths = paths
    st.session_state.uuids = predictions
    st.session_state.labels = [st.session_state.data['uuid_label_dic'][uuid] for uuid in predictions]
    # st.session_state.image_ind = 1
    show_images()
    # return paths, predictions


prompt = st.text_input("Prompt", "Why is the commercial not the same volume as the show uggh", on_change=get_templates, key="prompt")
image_ind = st.slider(label="Template ID", min_value=1, max_value=10, step=1, on_change=show_images, key="image_ind")

if "paths" not in st.session_state:
    get_templates()

ind = st.session_state.image_ind
st.image(st.session_state.img, caption=st.session_state.labels[ind-1].replace('-', " "))

# if not st.session_state.prompt_computed:
    # labels = [data['uuid_label_dic'][uuid] for uuid in uuids]


# @interact(k=kW, ind=images)

    # plt.axis("off")
    # plt.show()

# show_images(image_ind)
# file_name = paths[images.value-1]
# img = Image.open(os.path.join(DATA_PATH, file_name))
# img = img.convert(mode="RGB")

# uuid = uuids[images.value-1]
# gpt.prime_gpt_from_uuid(uuid)
# gpt_prompt = gpt.gpt.get_prime_text()
# label = data['uuid_label_dic'][uuid].replace("-", " ")
# prompt_begin = f"Give a humourous, witty meme caption based on the input provided. The label of this meme is '{label}'\n\n"
# gpt_prompt = prompt_begin + gpt_prompt + "input:" + prompt +"\noutput:"
# response = openai.Completion.create(
#   engine="text-davinci-002",
#   prompt=gpt_prompt,
#   temperature=temperature,
#   max_tokens=max_tokens,
#   frequency_penalty=frequency_penalty,
#   presence_penalty=presence_penalty
# )

# draw_caption_and_display(img, response)