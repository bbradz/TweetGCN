from __future__ import division, print_function
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from statsmodels.stats.contingency_tables import mcnemar
from model import TextGCN
from utils import load_all, build_graph

# Load and prepare data
labels, embeddings = load_all()
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Convert labels to one-hot for GCN if needed
train_labels_onehot = torch.nn.functional.one_hot(torch.tensor(y_train.astype(np.int64)), num_classes=np.unique(labels).size)
test_labels_onehot = torch.nn.functional.one_hot(torch.tensor(y_test.astype(np.int64)), num_classes=np.unique(labels).size)

# Convert embeddings to torch tensors
train_features = torch.tensor(X_train, dtype=torch.float32)
test_features = torch.tensor(X_test, dtype=torch.float32)

# Build graph adjacency matrices
train_adj = build_graph(train_features)
test_adj = build_graph(test_features)

# Convert labels to torch tensors (class indices)
train_labels_indices = torch.tensor(y_train, dtype=torch.int64)
test_labels_indices = torch.tensor(y_test, dtype=torch.int64)

# Initialize GCN Model
gcn_model = TextGCN(input_size=train_features.shape[1], num_classes=np.unique(labels).size)
optimizer = optim.Adam(gcn_model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()

# Training function
def train(epoch):
    gcn_model.train()
    optimizer.zero_grad()
    preds = gcn_model(train_features, train_adj)
    loss = loss_fn(preds, train_labels_indices)  # pass class indices instead of one-hot
    loss.backward()
    optimizer.step()
    return loss.item(), accuracy(preds, train_labels_indices).item()  # pass class indices to accuracy

# Test function
def test():
    gcn_model.eval()
    with torch.no_grad():
        preds = gcn_model(test_features, test_adj)
        return preds, loss_fn(preds, test_labels_indices).item(), accuracy(preds, test_labels_indices).item()  # pass class indices

# Fix accuracy function to handle indices directly
def accuracy(preds, actual):
    preds = torch.argmax(preds, dim=1)
    return (preds == actual).float().mean()

# Train and Evaluate GCN
for epoch in range(50):
    train_loss, train_acc = train(epoch)
    print(f'Epoch: {epoch+1}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}')

gcn_preds, test_loss, test_acc = test()
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

# Convert one-hot encoded predictions to label indices for comparison
gcn_predictions = torch.argmax(gcn_preds, dim=1).cpu().numpy()
predictions = {}
predictions['GCN'] = gcn_predictions

# Evaluate and Compare Models
models = [
    (LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000), "Logistic Regression"),
    (KNeighborsClassifier(), "KNN"),
    (SVC(), "SVM"),
    (XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), "XGBoost"),
    (DecisionTreeClassifier(), "Decision Tree")
]

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy:", round(accuracy, 4))

    predictions[model_name] = y_pred

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)
    print(f"{model_name} Mean FPR: ", np.mean(FPR))
    print(f"{model_name} Mean FNR: ", np.mean(FNR))
    print(" ")

for model, name in models:
    evaluate_model(model, X_train, y_train, X_test, y_test, name)

# Correct the comparison code to handle numpy array operations
def compare_models(predictions, y_test):
    model_names = list(predictions.keys())
    n_models = len(model_names)

    for i in range(n_models):
        for j in range(i + 1, n_models):
            print(f"Comparing {model_names[i]} vs {model_names[j]}:")
            preds_i = np.array(predictions[model_names[i]])
            preds_j = np.array(predictions[model_names[j]])
            for class_label in np.unique(y_test):
                is_class_i = (preds_i == class_label)
                is_class_j = (preds_j == class_label)
                not_class_i = (preds_i != class_label)
                not_class_j = (preds_j != class_label)

                b = np.sum(is_class_i & not_class_j)
                c = np.sum(not_class_i & is_class_j)
                if b + c > 0:
                    contingency_table = np.array([[0, b], [c, 0]])
                    result = mcnemar(contingency_table, exact=False)
                    print(f"Class {class_label} - Test statistic: {round(result.statistic, 4)}, p-value: {round(result.pvalue, 4)}")
                else:
                    print(f"Class {class_label} - Not enough discordant pairs to perform McNemar's test")

compare_models(predictions, y_test)
