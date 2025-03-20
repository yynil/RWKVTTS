import torch
import numpy as np
import sys
import os
import json
from sklearn.cluster import KMeans
jsonl_dir = sys.argv[1]
output_file_name = sys.argv[2]

# Load the embeddings from jsonl files the key is the name of the file
embeddings = {}
for file in os.listdir(jsonl_dir):
    print("Processing", file)
    if file.endswith("_embeddings.json"):
        with open(os.path.join(jsonl_dir, file), "r") as f:
            print("Loading", file)
            data = json.load(f)
            key_name = os.path.basename(file).replace("_embeddings.json", "")
            np_array = np.array(data)
            if np_array.shape[0] == 1:
                np_array = np_array[0]
            else:
                #find the cluster center of the embeddings using kmeans
                kmeans = KMeans(n_clusters=1, random_state=0, n_init = 'auto').fit(np_array)
                np_array = kmeans.cluster_centers_[0]
                
            embeddings[key_name]= {'embedding' : torch.tensor(np_array, dtype=torch.float32).unsqueeze(0)}
torch.save(embeddings, output_file_name)
print("Embeddings saved to", output_file_name)

state_dict = torch.load(output_file_name)
print("Loaded embeddings from", output_file_name)
for key in state_dict:
    print(key, state_dict[key]['embedding'].shape)