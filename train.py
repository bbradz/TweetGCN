from __future__ import division
from __future__ import print_function

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torch.nn.functional as F
import torch.optim as optim

import model
from utils import load_data, build_graph
from model import TextGCN

# Load data
train_features, train_labels, test_features, test_labels = load_data()

train_adj = build_graph(train_features)
test_adj = build_graph(test_features)

epoch_losses = []
epoch_accuracies = []
accuracy_per_label = {label: [] for label in range(6)}  # Assuming num_classes is defined
window_size = 100  # Define the window size for moving average
sliding_window_accuracies = []

# Model and optimizer
model = TextGCN(input_size=train_features.shape[1], num_classes=train_labels.shape[1])
optimizer = optim.Adam(model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()

def accuracy(preds, actual):

    actual = torch.argmax(actual, dim=1)
    preds = torch.argmax(preds, dim=1)
    preds = torch.flatten(preds)
    actual = torch.flatten(actual)

    n = actual.shape[0]
    # print("PRED: ")
    # print(preds)
    # print("======")
    # print("ACTUAL: ")
    # print(actual)
    # print("======")
    # print("ACC: ")
    # print(preds == actual)
    acc = (preds == actual).sum() / n

    return acc

def train(epoch):
    model.train()
    optimizer.zero_grad()

    preds = model(train_features, train_adj)

    training_loss = loss_fn(preds, train_labels)

    softmax = torch.nn.Softmax()
    preds = softmax(preds)

    training_accuracy = accuracy(preds, train_labels)

    epoch_losses.append(training_loss.item())
    epoch_accuracies.append(training_accuracy.item())

    # Compute sliding window average accuracy
    if epoch >= window_size:
        sliding_window_accuracy = np.mean(epoch_accuracies[-window_size:])
        sliding_window_accuracies.append(sliding_window_accuracy)

    training_loss.backward()
    optimizer.step()

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(training_loss.item()),
          'acc_train: {:.4f}'.format(training_accuracy.item()))


def test():
    model.eval()
    preds = model(test_features, test_adj)

    testing_loss = loss_fn(preds, test_labels)
    testing_accuracy = accuracy(preds, test_labels)

    # Convert tensor to numpy for t-SNE
    first_layer_embeddings_np = preds.detach().cpu().numpy()
    test_labels_np = test_labels.cpu().numpy()
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(first_layer_embeddings_np)
    print(embeddings_tsne.shape)
    print(embeddings_tsne[2939])
    labels_indices = np.argmax(test_labels_np, axis=1)

    # Plotting overall accuracy over time
    plt.figure(figsize=(10, 8))
    plt.plot(epoch_accuracies, label='Accuracy per Epoch')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plotting sliding window average accuracy
    plt.figure(figsize=(10, 8))
    plt.plot(sliding_window_accuracies, label='Sliding Window Average Accuracy')
    plt.title('Sliding Window Average Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    # Plot embeddings
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels_indices, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('t-SNE visualization of first layer GCN embeddings')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.show()

    print("Test set results:",
        "loss= {:.4f}".format(testing_loss.item()),
        "accuracy= {:.4f}".format(testing_accuracy.item()))


# Train model
for epoch in range(1_000):
    train(epoch)
print("Optimization Finished!")

# Testing
test()
