import numpy as np
import scipy.sparse as sp
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from preprocess import process_file

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def load_data(path="tweet_topic_single/dataset/split_coling2022_random/test_random.single.json", dataset="tweet_topic_single"):
    print('Loading {} dataset...'.format(dataset))

    output = process_file(path)
    features_str, labels_str = output[0], output[1]

    # Convert features and labels from string to correct types
    labels = encode_onehot(np.array(labels_str, dtype=np.float32))
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    features = model.encode(features_str)

    # Build symmetric adjacency matrix via cosine similarity of feature embeddings
    sim_matrix = cosine_similarity(features)
    adj = np.array(sim_matrix)

    # Normalize across each row
    adj_norm = adj / adj.sum(axis=1, keepdims=True)

    idx_train = range(1000)
    idx_val = range(1000, 2000)
    idx_test = range(2000, 3399)

    features = torch.FloatTensor(features) # Tensor of features
    labels = torch.LongTensor(np.where(labels)[1]) # Tensor of labels

    idx_train = torch.LongTensor(list(idx_train)) # Tensor of training indexes
    idx_val = torch.LongTensor(list(idx_val)) # Tensor of validation indexes
    idx_test = torch.LongTensor(list(idx_test)) # Tensor of testing indexes

    return adj_norm, features, labels, idx_train, idx_val, idx_test