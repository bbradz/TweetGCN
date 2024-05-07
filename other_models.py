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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.contingency_tables import mcnemar
from model import TextGCN
from utils import load_all, build_graph


# Calculate False Positive Rate and False Negative Ragte from predictions
def get_confusion(model_name, y_actual, y_pred, do_print=False):
    cm = confusion_matrix(y_actual, y_pred)
    if do_print:
        print(f"\n{cm}\n")

    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)
    f1 = f1_score(y_actual, y_pred, average='macro')

    print(f"{model_name} Mean FPR: ", round(np.mean(FPR), 4))
    print(f"{model_name} Mean FNR: ", round(np.mean(FNR), 4))
    print(f"{model_name} F1 Score: {round(f1, 4)}\n")

# Helper function for looping through sklearn model trainings and evaluations
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy:", round(accuracy, 4))

    predictions[model_name] = y_pred
    get_confusion(model_name, y_test, y_pred, False)

# Correct the comparison code to handle numpy array operations
def compare_models(predictions, y_test):
    model_names = list(predictions.keys())
    n_models = len(model_names)

    for i in range(n_models):
        for j in range(i + 1, n_models):
            print(f"{model_names[i]} vs {model_names[j]}:", end=" ")
            preds_i = np.array(predictions[model_names[i]])
            preds_j = np.array(predictions[model_names[j]])
            
            # Calculate b and c across all predictions
            b = np.sum((preds_i == y_test) & (preds_j != y_test))
            c = np.sum((preds_i != y_test) & (preds_j == y_test))
            
            # Perform McNemar's test if there are enough discordant pairs
            if b + c > 0:
                contingency_table = np.array([[0, b], [c, 0]])
                result = mcnemar(contingency_table, exact=False)
                print(result)
                print(f"-- Test statistic: {round(result.statistic, 4)}, p-value: {round(result.pvalue, 4)}")
            else:
                print("Not enough discordant pairs to perform McNemar's test")

# Load and prepare data
predictions = {}
labels, embeddings = load_all()
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Get one-hot labels for GCN and wrap labels in torch tensors
train_labels_onehot = torch.nn.functional.one_hot(torch.tensor(y_train.astype(np.int64)), num_classes=np.unique(labels).size)
test_labels_onehot = torch.nn.functional.one_hot(torch.tensor(y_test.astype(np.int64)), num_classes=np.unique(labels).size)
train_labels_indices = torch.tensor(y_train, dtype=torch.int64)
test_labels_indices = torch.tensor(y_test, dtype=torch.int64)

# Convert embeddings to torch tensors
train_features = torch.tensor(X_train, dtype=torch.float32)
test_features = torch.tensor(X_test, dtype=torch.float32)

# Build graph adjacency matrices
train_adj = build_graph(train_features)
test_adj = build_graph(test_features)

# Initialize GCN
gcn_model = TextGCN(input_size=train_features.shape[1], num_classes=np.unique(labels).size)
optimizer = optim.Adam(gcn_model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

# Train
for epoch in range(10_000):
    print(f"\r{epoch}", end="")
    gcn_model.train()
    optimizer.zero_grad()

    preds = gcn_model(train_features, train_adj)
    training_loss = loss_fn(preds, train_labels_indices)

    training_loss.backward()
    optimizer.step()


gcn_model.eval()
preds = gcn_model(test_features, test_adj)

softmax = torch.nn.Softmax(dim=1)
probs = softmax(preds)
predicted_labels = torch.argmax(probs, dim=1)
acc = (predicted_labels == test_labels_indices).float().mean()
print(f'\nGCN Accuracy: {round(acc.item(), 4)}')

# Convert one-hot encoded predictions to label indices for comparison
gcn_predictions = torch.argmax(preds, dim=1)
get_confusion("GCN", y_test, gcn_predictions, True)
predictions['GCN'] = gcn_predictions

# Evaluate and Compare Models
models = [
    (LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000), "Logistic Regression"),
    (KNeighborsClassifier(), "KNN"),
    (SVC(), "SVM"),
    (XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), "XGBoost"),
    (DecisionTreeClassifier(), "Decision Tree")
]

for model, name in models:
    evaluate_model(model, X_train, y_train, X_test, y_test, name)

compare_models(predictions, y_test)
