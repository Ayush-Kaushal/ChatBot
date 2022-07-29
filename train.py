import json
import numpy as np
from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))


ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
# print(all_words)
# print()
# print(tags)

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)  

X_train = np.array(X_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    

    def __len__(self):
        return self.n_samples


# Hyperparameters
batch_size = 8
hidden_size = 8
num_classes = len(tags)
input_size = len(X_train[0])
lr = 0.001
num_epochs = 500

dataset = ChatDataset()
train_loader = DataLoader(dataset, batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr)


# training

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)

        # forward
        outputs = model(words)
        loss = loss_fn(outputs, labels)

        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if (epoch + 1) % 100 == 0:
        print(f'epoch: {epoch + 1}/{num_epochs}, loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

# saving model

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": num_classes,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')