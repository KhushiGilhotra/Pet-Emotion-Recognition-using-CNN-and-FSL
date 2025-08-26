import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet34, ResNet34_Weights
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import random
from tqdm import tqdm

base_dir = r"C:\PETS_RESEARCH\pets_dataset\pets_dataset_split"
train_dir = os.path.join(base_dir, "new_train")
val_dir = os.path.join(base_dir, "new_valid")
test_dir = os.path.join(base_dir, "new_test")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

n_way = 3
n_shot = 5
n_query = 10
n_val_query = 15
n_episodes = 600

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

train_dataset = ImageFolder(train_dir, transform=transform)
val_dataset = ImageFolder(val_dir, transform=transform)
test_dataset = ImageFolder(test_dir, transform=transform)

class_names = train_dataset.classes
print("Detected classes:", class_names)

class ProtoNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet34(weights=ResNet34_Weights.DEFAULT)
        base_model.fc = nn.Identity()
        self.encoder = base_model

    def forward(self, x):
        return self.encoder(x)

def create_episode(dataset, n_way, n_shot, n_query):
    class_to_indices = {i: [] for i in range(len(dataset.classes))}
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices[label].append(idx)

    selected_classes = random.sample(sorted(class_to_indices.keys()), n_way)
    support_indices, query_indices = [], []

    for cls in selected_classes:
        samples = random.sample(class_to_indices[cls], n_shot + n_query)
        support_indices += samples[:n_shot]
        query_indices += samples[n_shot:]

    support = [dataset[i] for i in support_indices]
    query = [dataset[i] for i in query_indices]

    s_x = torch.stack([x for x, _ in support])
    s_y = torch.tensor([selected_classes.index(y) for _, y in support])
    q_x = torch.stack([x for x, _ in query])
    q_y = torch.tensor([selected_classes.index(y) for _, y in query])

    return s_x, s_y, q_x, q_y

def prototypical_loss(embeddings, targets, n_way, n_shot, n_query, use_cosine=False):
    support = embeddings[:n_way * n_shot]
    query = embeddings[n_way * n_shot:]

    support = support.view(n_way, n_shot, -1)
    prototypes = support.mean(dim=1)

    if use_cosine:
        query_norm = F.normalize(query, dim=1)
        proto_norm = F.normalize(prototypes, dim=1)
        logits = torch.matmul(query_norm, proto_norm.T)
    else:
        dists = torch.cdist(query, prototypes)
        logits = -dists

    loss = F.cross_entropy(logits, targets)
    preds = torch.argmax(logits, dim=1)
    acc = (preds == targets).float().mean().item()

    return loss, acc

model = ProtoNetBackbone().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_val_acc = 0

print("\nStarting training...\n")
for episode in range(1, n_episodes + 1):
    model.train()
    s_x, s_y, q_x, q_y = create_episode(train_dataset, n_way, n_shot, n_query)
    s_x, q_x, s_y, q_y = s_x.to(device), q_x.to(device), s_y.to(device), q_y.to(device)

    optimizer.zero_grad()
    embeddings = model(torch.cat([s_x, q_x], dim=0))
    loss, acc = prototypical_loss(embeddings, q_y, n_way, n_shot, n_query)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        s_x, s_y, q_x, q_y = create_episode(val_dataset, n_way, n_shot, n_val_query)
        s_x, q_x, s_y, q_y = s_x.to(device), q_x.to(device), s_y.to(device), q_y.to(device)
        embeddings = model(torch.cat([s_x, q_x], dim=0))
        _, val_acc = prototypical_loss(embeddings, q_y, n_way, n_shot, n_val_query)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "c:/PETS_RESEARCH/PetMoodDetector/best_protonet.pth")

    if episode % 10 == 0:
        print(f"Episode {episode}/{n_episodes} - Train Acc: {acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")


print("\n✅ Final Test Evaluation...\n")
model.load_state_dict(torch.load("c:/PETS_RESEARCH/PetMoodDetector/best_protonet.pth"))
model.eval()

all_preds, all_labels = [], []

for _ in range(100):
    s_x, s_y, q_x, q_y = create_episode(test_dataset, n_way, n_shot, n_query)
    s_x, q_x = s_x.to(device), q_x.to(device)
    embeddings = model(torch.cat([s_x, q_x], dim=0))
    support = embeddings[:n_way * n_shot].view(n_way, n_shot, -1).mean(dim=1)
    query = embeddings[n_way * n_shot:]

    dists = torch.cdist(query, support)
    preds = torch.argmin(dists, dim=1)
    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(q_y.numpy())  

print(f"✅ Final Test Accuracy over 100 episodes: {(np.mean(np.array(all_preds) == np.array(all_labels)) * 100):.2f}%")
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
    balanced_accuracy_score
)
import numpy as np
import torch

model.load_state_dict(torch.load("c:/PETS_RESEARCH/PetMoodDetector/best_protonet.pth"))
model.eval()

NUM_TEST_EPISODES = 100  
all_preds = []
all_labels = []
per_episode_stats = []

with torch.no_grad():
    for ep in range(NUM_TEST_EPISODES):
        s_x, s_y, q_x, q_y = create_episode(test_dataset, n_way, n_shot, n_query)

        s_x = s_x.to(device)
        q_x = q_x.to(device)

        embeddings = model(torch.cat([s_x, q_x], dim=0))

        support_emb = embeddings[: n_way * n_shot].view(n_way, n_shot, -1).mean(dim=1)  
        query_emb = embeddings[n_way * n_shot :]  

        dists = torch.cdist(query_emb, support_emb)            
        preds = torch.argmin(dists, dim=1).cpu().numpy()       

        labels = q_y.cpu().numpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

        ep_acc = (preds == labels).mean()
        ep_prec, ep_rec, ep_f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
        per_episode_stats.append((ep_acc, ep_prec, ep_rec, ep_f1))

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[Episode {ep+1}/{NUM_TEST_EPISODES}] Acc: {ep_acc*100:.2f}%, Prec(macro): {ep_prec:.4f}, Rec(macro): {ep_rec:.4f}, F1(macro): {ep_f1:.4f}")

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

overall_acc = accuracy_score(all_labels, all_preds)
balanced_acc = balanced_accuracy_score(all_labels, all_preds)

prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro', zero_division=0)
prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)

per_class_prec, per_class_rec, per_class_f1, per_class_support = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)

report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)

cm = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-12)

print("\n===== Aggregate Test Results (over {} episodes) =====".format(NUM_TEST_EPISODES))
print(f"Overall Accuracy       : {overall_acc*100:.2f}%")
print(f"Balanced Accuracy      : {balanced_acc*100:.2f}%")
print(f"Precision (macro)      : {prec_macro:.4f}")
print(f"Recall (macro)         : {rec_macro:.4f}")
print(f"F1-score (macro)       : {f1_macro:.4f}")
print(f"Precision (micro)      : {prec_micro:.4f}")
print(f"Recall (micro)         : {rec_micro:.4f}")
print(f"F1-score (micro)       : {f1_micro:.4f}")
print(f"Precision (weighted)   : {prec_weighted:.4f}")
print(f"Recall (weighted)      : {rec_weighted:.4f}")
print(f"F1-score (weighted)    : {f1_weighted:.4f}")

print("\n--- Per-class metrics ---")
for i, cname in enumerate(class_names):
    print(f"{cname:15s} | Prec: {per_class_prec[i]:.4f} | Rec: {per_class_rec[i]:.4f} | F1: {per_class_f1[i]:.4f} | Support: {per_class_support[i]}")

print("\n--- Classification Report ---")
print(report)

print("\n--- Confusion Matrix (raw) ---")
print(cm)
print("\n--- Confusion Matrix (normalized by true class rows) ---")
print(np.round(cm_norm, 3))

with open("c:/PETS_RESEARCH/PetMoodDetector/test_classification_report.txt", "w") as f:
    f.write("Aggregate Test Results\n")
    f.write(f"Overall Accuracy: {overall_acc*100:.2f}%\n\n")
    f.write(report)
    f.write("\nConfusion Matrix (raw):\n")
    np.savetxt(f, cm, fmt='%d')
    f.write("\nConfusion Matrix (normalized):\n")
    np.savetxt(f, np.round(cm_norm,3), fmt='%.3f')

print("\nSaved classification report to: c:/PETS_RESEARCH/PetMoodDetector/test_classification_report.txt")
