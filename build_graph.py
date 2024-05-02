import pandas as pd
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# loading dataset
# TODO: this assumes csv structure with text column
df = pd.read_csv('data.json')
texts = df['text'].tolist()

# load sentence bert
model = SentenceTransformer('all-MiniLM-L6-v2')

# generate embeddings
embeddings = model.encode(texts, show_progress_bar=True)

# generate cosine similarities
similarity_matrix = cosine_similarity(embeddings)

# construct fully connected graph
G = nx.Graph()
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        # Add an edge between every pair of nodes with the cosine similarity score as the weight
        G.add_edge(i, j, weight=similarity_matrix[i][j])

# printing for sanity check
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

# save graph
nx.write_weighted_edgelist(G, "graph.edgelist")
