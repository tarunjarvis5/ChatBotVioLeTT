import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pre_processor import bag_of_words, tokenizer, stem_it, pre_processor_engine
from model import NeuralNet

#open training data
with open('finData.json', 'r') as f:
        intents = json.load(f)



# get preprocessed data
processed_data = pre_processor_engine(intents)

all_words = processed_data["all_words"] 
tags = processed_data["tags"] 
xy = processed_data["xy"] 
X_train = processed_data["X_train"] 
y_train = processed_data["y_train"] 


# Hyper-parameters 
num_epochs = 1000
batch_size = 8       # number of samples processed before we change the model
learning_rate = 0.001    # 
input_size = len(X_train[0])   # size of bag of words
hidden_size = 8      
output_size = len(tags)      
print(input_size, output_size)


# create torch dataset
class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # print loss every 100 epoch
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# total loss in the model
print(f'final loss: {loss.item():.4f}')

# store all the trained attributes and pickle into a file and use for chat
data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)


