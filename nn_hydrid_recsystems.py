import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import time
import pickle

"""
This code was adapted largely from:
https://github.com/AIPI540/AIPI540-Deep-Learning-Applications/blob/main/4_recommenders/nn_hybrid_recommender.ipynb
"""

# load data
df = pd.read_csv('data/00_raw/transactions_train.csv')
articles = pd.read_csv('data/00_raw/articles.csv')
articles = articles[['article_id', 'prod_name', 'colour_group_name']]
df = df.merge(articles, how='left', on='article_id')

# create unbought data
df2 = pd.DataFrame()
df2['customer_id'] = df['customer_id'].sample(n=len(df))
df2['prod_name'] = df['prod_name'].sample(n=len(df))
df2['total_bought'] = 0

# create model train df
data = df.groupby(['customer_id', 'prod_name'])['customer_id'].count().reset_index(name='total_bought')
data = pd.concat([data, df2]).reset_index(drop=True)
data['total_bought'] = np.where(data['total_bought'] > 2, 2, data['total_bought'])
color_join = df.drop_duplicates(subset=['customer_id', 'prod_name'])
data = data.merge(color_join, how='left', on=['customer_id', 'prod_name'])

# Encode the customer data
encoder = LabelEncoder()
encoder.fit(data['customer_id'])
data['encoded_customer_id'] = encoder.transform(data['customer_id'])

# Encode the product data
encoder = LabelEncoder()
encoder.fit(data['prod_name'])
data['encoded_prod_name'] = encoder.transform(data['prod_name'])

# Encode the color data
encoder = LabelEncoder()
encoder.fit(data['colour_group_name'])
data['encoded_colour_group_name'] = encoder.transform(data['colour_group_name'])

X = data.loc[:,['encoded_customer_id','encoded_prod_name','encoded_colour_group_name']]
y = data.loc[:,'total_bought']

# Split our data into training and test sets
X_train, X_val, y_train, y_val = train_test_split(X,y,random_state=0, test_size=0.05)

def prep_dataloaders(X_train,y_train,X_val,y_val,batch_size):
    # Convert training and test data to TensorDatasets
    trainset = TensorDataset(torch.from_numpy(np.array(X_train)).long(), 
                            torch.from_numpy(np.array(y_train)).float())
    valset = TensorDataset(torch.from_numpy(np.array(X_val)).long(), 
                            torch.from_numpy(np.array(y_val)).float())

    # Create Dataloaders for our training and test data to allow us to iterate over minibatches 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader

batchsize = 64
trainloader,valloader = prep_dataloaders(X_train,y_train,X_val,y_val,batchsize)

class NNHybridFiltering(nn.Module):
    
    def __init__(self, n_users, n_items, n_colors, embdim_users, embdim_items, embdim_colors, n_activations, rating_range):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_embeddings=n_users,embedding_dim=embdim_users)
        self.item_embeddings = nn.Embedding(num_embeddings=n_items,embedding_dim=embdim_items)
        self.color_embeddings = nn.Embedding(num_embeddings=n_colors,embedding_dim=embdim_colors)
        self.fc1 = nn.Linear(embdim_users+embdim_items+embdim_colors,n_activations)
        self.fc2 = nn.Linear(n_activations,1)
        self.rating_range = rating_range

    def forward(self, X):
        # Get embeddings for minibatch
        embedded_users = self.user_embeddings(X[:,0])
        embedded_items = self.item_embeddings(X[:,1])
        embedded_colors = self.color_embeddings(X[:,2])
        # Concatenate user, item and color embeddings
        embeddings = torch.cat([embedded_users,embedded_items,embedded_colors],dim=1)
        # Pass embeddings through network
        preds = self.fc1(embeddings)
        preds = F.relu(preds)
        preds = self.fc2(preds)
        # Scale predicted ratings to target-range [low,high]
        preds = torch.sigmoid(preds) * (self.rating_range[1]-self.rating_range[0]) + self.rating_range[0]
        return preds

def train_model(model, criterion, optimizer, dataloaders, device, num_epochs=5, scheduler=None):
    model = model.to(device) # Send model to GPU if available
    since = time.time()

    costpaths = {'train':[],'val':[]}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Get the inputs and labels, and send to GPU if available
            for (inputs,labels) in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the weight gradients
                optimizer.zero_grad()

                # Forward pass to get outputs and calculate loss
                # Track gradient only for training data
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model.forward(inputs).view(-1)
                    loss = criterion(outputs, labels)

                    # Backpropagation to get the gradients with respect to each weight
                    # Only if in train
                    if phase == 'train':
                        loss.backward()
                        # Update the weights
                        optimizer.step()

                # Convert loss into a scalar and add it to running_loss
                running_loss += np.sqrt(loss.item()) * labels.size(0)

            # Step along learning rate scheduler when in train
            if (phase == 'train') and (scheduler is not None):
                scheduler.step()

            # Calculate and display average loss and accuracy for the epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            costpaths[phase].append(epoch_loss)
            print('{} loss: {:.4f}'.format(phase, epoch_loss))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return costpaths

# Train the model
dataloaders = {'train':trainloader, 'val':valloader}
n_users = X.loc[:,'encoded_customer_id'].max()+1
n_items = X.loc[:,'encoded_prod_name'].max()+1
n_colors = X.loc[:,'encoded_colour_group_name'].max()+1
model = NNHybridFiltering(n_users,
                       n_items,
                       n_colors,
                       embdim_users=50, 
                       embdim_items=50, 
                       embdim_colors=25,
                       n_activations = 100,
                       rating_range=[0., 2.])
criterion = nn.MSELoss()
lr=0.001
n_epochs=10
wd=1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
device = torch.device("cuda")

cost_paths = train_model(model,criterion,optimizer,dataloaders, device,n_epochs, scheduler=None)

def predict_rating(model, userId, movieId, color, encoder, device):
    # Encode color
    color = encoder.transform(np.array(color).reshape(-1))
    # Get predicted rating
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        X = torch.Tensor([userId,movieId,color]).long().view(1,-1)
        X = X.to(device)
        pred = model.forward(X)
        return pred

# Get predictions for all users 
test_customers = list(X_val.encoded_customer_id.unique())
test_prod = list(data.encoded_prod_name.unique())
colors = list(data.colour_group_name.unique())

recommendations = {}
for customer in test_customers:
  customer_recs = {}
  prod_color = {}
  for prod in test_prod:
    init_rating = 0
    prod_col = None
    for color in colors:
        rating = predict_rating(model,userId=customer,movieId=prod,color=color,encoder=encoder, device=device)
        if rating >= init_rating:
            init_rating = rating
            top_color = color
    customer_recs[prod] = init_rating
    prod_color[prod] = top_color
  sorted_customer_recs = {k: v for k, v in sorted(customer_recs.items(), key=lambda item: item[1])}
  recs = list(sorted_customer_recs.keys())[:12]
  recommendations[customer] = recs

# output recommendations in pickle form
pickle.dump(recommendations, open("models/nn_rec.pickle", "wb"))