'''
Refactor annotated_qa folder to merge all the user prompts, meme caption pairs into one big pkl file. This is better
at inference time
'''

from pathlib import Path
import pickle

folder_path = Path("../data/annotated_qa")
files = []
manual_files = []
for file in folder_path.glob("*.pkl"):
    if "_manual" not in str(file):
        files.append(file)

for file in files:
    manual_file = str(file).replace(".pkl", "_manual.pkl")
    manual_files.append(manual_file)

dictions = []
manuals = []
for file in files:
    with open(file, 'rb') as f:
        diction = pickle.load(f)
        dictions.append(diction)

for file in manual_files:
    with open(file, 'rb') as f:
        diction = pickle.load(f)
        manuals.append(list(diction.values()))

final_dicts = dict()
errors = 0
for manual_file, dict1 in zip(manuals, dictions):
    uuid = dict1['uuid']
    tuples = []
    for user_prompt in manual_file:
        if len(user_prompt):
            tuples.append((user_prompt, dict1['qa'][user_prompt]))
    final_dicts[uuid] = tuples

with open("../data/gpt3_user_prompt_dic.pkl", 'wb') as f:
    pickle.dump(final_dicts, f)
