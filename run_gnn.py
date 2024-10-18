#%%from helpers import *
from gnn import *
import os
import numpy as np
import warnings
from sklearn.utils import shuffle
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GeoLoader
import gpytorch
from helpers import *
seed = 42
import torch.nn.functional as F
import random
import re
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
#%%

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device used: {device}')
graph = []
#create a dummy variable that will serve as the prediction target, i.e.
#let us assume that the e_value is the value of the energy of the entire molecule
data_list = []
root_path = os.getcwd() + '/data_processed/'
filenames = os.listdir(root_path)
filenames.sort()
# debug: 
# test_string = 'data_1.pt'
# find_first = re.compile('_1.pt')
counter_for_cluster = 0
counter_boring = 0
# filenames = filenames[:6]
for filename in filenames:
    if filename.endswith('.pt'):
        # print(filename)
        graph = torch.load(os.path.join(root_path, filename)) 
        graph.x = graph.x.to(device)
        # print('x', graph.x) 
        graph.edge_attr = graph.edge_attr.to(device)
        # print('attr', graph.edge_attr)
        graph.edge_index = graph.edge_index.to(device)
        # print('index', graph.edge_index)
        graph.y = torch.tensor(graph.y).to(device)
        data_list.append(graph)  

#%%
# we choose wether the computations will be on the gpu or the cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)

train_loader = GeoLoader(train_data, batch_size=len(data_list), shuffle=True)
test_loader = GeoLoader(test_data, batch_size=len(test_data), shuffle=True)

gnn_model = BaseGNN(data_list[0].x.shape[-1], data_list[0].edge_attr.shape[-1])
gnn_model.to(device)

learning_rate = 0.01
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.2, patience=10,
                                                       min_lr=0.0000001)
criterion = F.mse_loss
# %%
############################## CASUAL TRAINING TEST SETUP ##############################
## the training loop##
for epoch in range(1000):
    lr = scheduler.optimizer.param_groups[0]['lr']
    gnn_model.train()
   
    for data in train_loader:
        data = data.to(device)
        
        optimizer.zero_grad()
        
        loss = criterion(gnn_model(data).float(), data.y.float())
    
        loss.backward()
        
        optimizer.step()

        scheduler.step(loss)

    # Testing loop
    gnn_model.eval()

    with torch.no_grad():  
        for data in test_loader:
            data = data.to(device)
            
            output = gnn_model(data).float()
            
            test_loss = criterion(output, data.y.float())
            print('Testing loss', test_loss.item())

    
    average_test_loss = test_loss / len(test_loader)
    print('Average testing loss:', average_test_loss.item())

#%%
############################## CROSS VALIDATION ##############################
sampling_size = 3
data_len = len(data_list)

i = 0
test_pred = []
test_truth = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_batches = data_len // sampling_size
indices = np.arange(data_len)
for i in range(num_batches):
    best_val_loss = np.inf
    
    start_index = i * sampling_size

    val_test_indices = indices[start_index:start_index + sampling_size]
    np.random.shuffle(val_test_indices)
    
    val_indices = [val_test_indices[2]]
    test_indices = val_test_indices[:2]
    train_indices = np.setdiff1d(indices, val_test_indices)
     
    train_data = [data_list[i] for i in train_indices]
    val_data = [data_list[i] for i in val_indices]
    test_data = [data_list[i] for i in test_indices]

    train_loader = GeoLoader(train_data, batch_size=len(train_data), shuffle=True)
    val_loader = GeoLoader(val_data, batch_size=len(val_data), shuffle=False)
    test_loader = GeoLoader(test_data, batch_size=len(test_data), shuffle=False)

    gnn_model = BaseGNN(data_list[0].x.shape[-1], data_list[0].edge_attr.shape[-1])
    gnn_model.to(device)
 
    learning_rate = 0.01
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.2, patience=10,
                                                       min_lr=0.0001)
    criterion = F.mse_loss
    for epoch in range(1000):
        lr = scheduler.optimizer.param_groups[0]['lr']
        gnn_model.train()

        for data in train_loader:
           
            data = data.to(device)        
            optimizer.zero_grad()

            loss = criterion(gnn_model(data).float(), data.y.float())
            
            loss.backward()

            optimizer.step()
            
            scheduler.step(loss)

        val_loss = 0.0
        test_loss = 0.0

        gnn_model.eval()
        with torch.no_grad():  
            for data in val_loader:
                data = data.to(device)
                
                output = gnn_model(data).float()
                
                val_loss = criterion(output, data.y.float())

        gnn_model.eval()
        with torch.no_grad():  
            for data in test_loader:
                data = data.to(device)
                
                test_output = gnn_model(data).float()
                
                test_loss = criterion(output, data.y.float())
        if epoch == range(1000)[-1]:
            # print(f'Epoch {epoch}:Training loss: {loss},  Val loss: {val_loss.item()}, Test loss: {test_loss.item()}')
            print(f'Split {i}: Training loss: {loss},  Val loss: {val_loss.item()}, Test loss: {test_loss.item()}')

        val_loss /= len(val_loader)
        test_loss /= len(test_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_loss = test_loss
            best_model_state = gnn_model.state_dict()
            pred_temp = test_output
            data_temp = data.y
            
    test_pred.append(pred_temp)
    test_truth.append(data_temp)
print('testpred', test_pred)
print('testtruth', test_truth)
test_pred = torch.cat(test_pred).numpy()
test_truth = torch.cat(test_truth).numpy()
test_truth = test_truth.reshape(-1,1)

pred_truth_se =np.array([(test_pred[i] - test_truth[i])**2 for i in range(test_truth.shape[0])])
# print('se', pred_truth_se)
# print('se shape', pred_truth_se.shape)
pred_truth_se_mean = np.mean(pred_truth_se)
pred_truth_se_std = np.std(pred_truth_se)
print('mean', pred_truth_se_mean)
print('std' ,pred_truth_se_std)

small_errors =[] 
for i in pred_truth_se:
    diff = i - pred_truth_se_mean
    if abs(diff)< pred_truth_se_std:
        small_errors.append(i)
small_errors = np.array(small_errors)
# print('small errors', small_errors) 
# print('small errors shape', small_errors.shape)    

#%%
############################## PLOTTING FOR CROSS VALIDATIONN ##############################
import matplotlib.pyplot as plt 
figsize = (12, 4)
fig, axs = plt.subplots(1,3, figsize = figsize)
axs.flatten()
axs[0].scatter(test_truth, test_pred, color='blue', label='Predictions vs Actual')
axs[0].plot([min(test_truth), max(test_truth)], [min(test_truth), max(test_truth)], color='red', linestyle='--', label='y = x')
axs[0].set_xlabel('Actual Labels')
axs[0].set_ylabel('Predictions')
axs[0].set_title('Predictions vs Actual')

axs[1].bar(range(len(pred_truth_se)), pred_truth_se.flatten())
axs[1].set_xlabel('x')
axs[1].set_ylabel('Squared Error')
axs[1].set_title('All errors')

axs[2].bar(range(len(small_errors)), small_errors.flatten())
axs[2].set_xlabel('x')
axs[2].set_ylabel('Squared Error')
axs[2].set_title('Only small errors')

plt.tight_layout()
plt.show()
# %%    
## prediction time##
# same layers that behave different during training behave different when we want to use the 
#network for prediction so we switch them to .eval() to generate predictions
gnn_model.eval()

# we could wrap the data into another loader but there is another way which you see below
pred = gnn_model(Batch.from_data_list(test_data).to(device))
print(pred)

# %%
import matplotlib.pyplot as plt 
final_test_pred = pred.cpu().detach().numpy()
actual_labels = [data.y.detach().cpu().numpy() for data in test_data]

print('actual labels', actual_labels)
print('test pred', final_test_pred)

print(np.array(test_data).shape)
plt.scatter(actual_labels, final_test_pred, color='blue', label='Predictions vs Actual')
plt.plot([min(actual_labels), max(actual_labels)], [min(actual_labels), max(actual_labels)], color='red', linestyle='--', label='y = x')
# plt.plot(test_data, pred)
plt.xlabel('Actual Labels')
plt.ylabel('Predictions')
plt.title('Predictions vs Actual')
plt.legend()
plt.show()
# %%
hyperparameter_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'hidden_size': [64, 128, 256],
    'num_layers': [1, 2, 3],
}

# Create all possible combinations of hyperparameters
param_combinations = list(ParameterGrid(hyperparameter_grid))
print(len(param_combinations))
# %%
one = np.random.rand(14,1)
two = np.random.rand(14,1)
res = [(one[i] - two[i])**2 for i in range(one.shape[0])]
print(np.array(res).shape)


# %%
import matplotlib.pyplot as plt

# Example list of 14 values
data = [2, 5, 8, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35, 38]

# Create a figure and axis
fig, axs = plt.subplots()

# Plotting the bars using axs.bar
axs.bar(range(len(data)), data, color='blue', edgecolor='black', alpha=0.7)

# Adding labels and title
axs.set_xlabel('Index')
axs.set_ylabel('Values')
axs.set_title('Bar Plot of Values')

# Show the plot
plt.show()

# %%

x1 = np.linspace(1, 14, 14)
print(type(i) for i in x1)
x2 = range(1, 13+1)
print(x1)
print(type(range(len(x1))))
# %%
