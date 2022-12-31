# <img src="https://user-images.githubusercontent.com/18485647/205524790-25d7b702-3bbc-4920-b586-ea25214ad99f.png" width="30" height="30"/> JESTER: A SOTA Text to Meme Generation Model
[Web Demo](https://the-jester.streamlit.app/) | [Report](https://github.com/SumanthRH/text-to-meme/blob/main/Final_Report.pdf)

We propose JESTER, a text-to-meme generation engine. Enter any text and get a relevant meme in seconds, all from your browser! 

### Methodology
JESTER consists of two parts as shown in the figure below:

1. A meme template retrieval system which use RoBERTa model finetuned using Contrastive Loss. The model is trained to create meme embeddings in a 
high-dimensional landscape that capture the inherent sarcasm and language context of a meme. The model retrives the templates whose embeddings are 
closest to the user input embedding by cosine similarity scores.

2. A caption generation system that uses GPT-3 to generate creative caption of the meme based on the user input and the meme template context. 
A few manually labeled examples are sent as QA pairs along with the user input as the prompt to GPT-3.

<img width="708" alt="Screen Shot 2022-12-04 at 4 23 23 PM" src="https://user-images.githubusercontent.com/18485647/205525021-dba35931-d3ea-4f30-bd34-2aaa26392343.png">

### Output Samples

In the figure below, we show some sample generated using JESTER. Both the input user prompt and the final generated image are shown.

<img width="697" alt="Screen Shot 2022-12-04 at 4 27 18 PM" src="https://user-images.githubusercontent.com/18485647/205525266-2e8e0b46-660d-4116-a362-8ee77ead9660.png">


### Dataset

We use the [Deep Humour](https://github.com/ilya16/deephumor) dataset. Due to limited computational budget, we restrict ourselves to only 100 templates, with a total of 300,000 captions. Since we're running our Streamlit app right from GitHub, we've put all the data on the repo in the `data\` folder. The cleaned and preprocessed data used for training is the `data/meme_900k_cleaned_data_v2.pkl` file. We make use of UUIDs for addressing each template. The `pkl` file contains the following dictionaries:

| Dictionary | Mapping |
| ----------- | -------|
| `label_uuid_dic`| Template Label (like "not sure if") to UUID|
| `uuid_label_dic` | UUID to Template Tabel|
| `uuid_caption_dic` | UUID to List of Captions (for that template)|
| `uuid_image_path_dic` | UUID to Template Image Path |

### Training [![Open in GitHub](https://img.shields.io/badge/_-Open_in_GitHub-blue.svg?logo=Jupyter&labelColor=5c5c5c)](notebooks/sentencebert-finetuning.ipynb)

If you wish the train the model yourselves, you can use the following notebooks:
* `notebooks/transformer_training.ipynb` : Finetuning a softmax-based vanilla RoBERTa model 
* `notebooks/sentencebert-finetuning.ipynb`: Training the Sentence-RoBERTa model (used in the final demo) based on Contrastive Loss. 


The final template embeddings are all stored in `pkl` files in `models/model_utils` . We use Git-LFS to store the model checkpoints, referenced at `model/sentence_transformer_roberta_samples_100_epochs_5/'. 

#### Notebook Demo [![Open in GitHub](https://img.shields.io/badge/_-Open_in_GitHub-blue.svg?logo=Jupyter&labelColor=5c5c5c)](notebooks/Final-Demo.ipynb)
For some reason (Why??) if you wish to use a notebook demo instead of the [web demo](https://the-jester.streamlit.app/), that's available at `notebooks/Final-Demo.ipynb`. 

### Sensitive content
Memes push the boundaries of what is comfortable. In every dataset we sought to use, including the [Deep Humour](https://github.com/ilya16/deephumor) dataset, there was a significant amount of hate speech in various forms. It was simply impossible for us to completely filter out these datasets. We've thus implemented some safeguards at two levels: 1) All the labelled examples we feed into GPT-3 has been carefully chosen to weed out sensitive content 2) We have implemented a check up-front to detect hate speech in user prompts using [hatesonar](https://github.com/Hironsan/HateSonar).

### Terms of Use

The model should not be used to spread messages/ ideas that in any way is unlawful, defamatory, obscene, or otherwise objectionable. 






