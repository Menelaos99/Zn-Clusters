#%%
import os
import re
import numpy as np
import warnings
from sklearn.utils import shuffle
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GeoLoader
import gpytorch
from helpers import get_graph
from models import BaseGNN, ExactGPModel
import torch.nn.functional as F
seed = 42
import random
import itertools
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
import torch.nn as nn
warnings.filterwarnings("ignore", category=RuntimeWarning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%
def data_preparation(folder='/data_processed/', device=device, types_to_include=['Pt3', 'Pt6', 'Pt9']):
    data_list = []
    
    data_dict = {t: {} for t in types_to_include}
    root_path = os.getcwd() + folder
    filenames = os.listdir(root_path)
    filenames.sort()
    pattern = re.compile(r'(.*)_(\d+)_data_(-?\d+)\.pt')

    for filename in filenames:
        if filename.endswith('.pt'):
            match = pattern.match(filename)
            if match:
                type_key, file_index, data_index = match.groups()
                file_index = int(file_index) 
                data_index = int(data_index)  

                if type_key in types_to_include:
                    full_path = os.path.join(root_path, filename)
                    graph = torch.load(full_path)
                    graph.x = graph.x.to(device)
                    graph.edge_attr = graph.edge_attr.to(device)
                    graph.edge_index = graph.edge_index.to(device)
                    graph.y = torch.tensor(graph.y, dtype=torch.float).to(device)
                    data_list.append(graph)

                    if file_index not in data_dict[type_key]:
                        data_dict[type_key][file_index] = {}
                    if data_index not in data_dict[type_key][file_index]:
                        data_dict[type_key][file_index][data_index] = []
                    data_dict[type_key][file_index][data_index].append(graph)
    
    return data_list, data_dict


def atom_extraction(cluster_name = 'Pt3'):

    with open('data_processed/' + cluster_name + '_atoms.txt', 'r') as f:
        lines = f.readlines()
        atoms = [line.strip() for line in lines]
    return atoms


def train(train_loader,optimizer,device, model_gp, mll, unmasked_data_list, domains):
    # "Loss" for GPs - the marginal log likelihood
    for data in train_loader:
        domain_means = []
        data = data.to(device)
        # zero gradients from previous iteration
        optimizer.zero_grad()
        # output from model
        output = model_gp(data)

        means = output.mean.detach().cpu().clone().numpy()
        std_print = output.stddev.detach().cpu().clone().numpy()
        
        domain_means = []
        unique_domains = np.unique(np.array(domains))
        # indices = np.argwhere(np.isin(domains, unique_domains))

        for unique in unique_domains:
            mean_extraction =[]
            for i in range(len(domains)):
                if unique == domains[i]:
                    mean_extraction.append(means[i])
            domain_means.append(mean_extraction)
        std = 0
        
        for i in range(len(unique_domains)):
            std += np.std(np.array(domain_means[i]))
        # a = torch.tensor(np.histogram(np.array([0.1, 0.2, 0.3]))[-1])
        # b = torch.tensor(np.histogram(np.array([0.2, 0.4]))[-1])
        # loss = nn.KLDivLoss()
        # dist = loss(a, b)
        unique_pairs = list(itertools.combinations(list(range(0, len(unique_domains))), 2))
        shannon_div = 0
        for k,j in unique_pairs:
            q = np.array(domain_means[k])
            p = np.array(domain_means[j])

            shape_q = q.shape
            shape_p = p.shape

            qs = np.concatenate([q.copy() for _ in range(shape_p[0])])            
            ps = np.concatenate([p.copy() for _ in range(shape_q[0])])            
            shannon_div += np.sum(qs * np.log(qs / ps)) +np.sum(ps * np.log(ps / qs))
        # for index in range(indices.shape[0]):
        #     print('idxlen', means[index])
        #     domain_means.append(means[index])
        # domain_means = np.array(domain_means)      

        # calc loss and backprop gradients
        if np.isnan(shannon_div):
            shannon_div = 0
        loss = -mll(output, torch.tensor(data.y, dtype=torch.float).to(device).reshape(-1,))# + std - shannon_div
        #print('total loss', loss)
        #print('contrastive', -std + shannon_div)
        #print('std', -std )
        #print('shannon', shannon_div)
        #print('MLL', -mll(output, torch.tensor(data.y, dtype=torch.float).to(device).reshape(-1,)))
        loss.backward()
        optimizer.step()
    return loss / len(train_loader.dataset), means, std_print

def validate(validation_loader, device, model_gp, likelihood, mll):
    model_gp.eval()
    likelihood.eval()

    with torch.no_grad():
        validation_loss = 0.0
        for data in validation_loader:
            data = data.to(device)
            # output = likelihood(model_gp(data))
            output = model_gp(data)

            loss = -mll(output, torch.tensor(data.y, dtype=torch.float).to(device).reshape(-1,))
            validation_loss += loss.item()
        validation_loss /= len(validation_loader.dataset)
    return validation_loss, output, data.y

def test(test_loader, device, model_gp, likelihood, mll):
    model_gp.eval()
    likelihood.eval()

    with torch.no_grad():
        test_loss = 0.0
        for data in test_loader:
            data = data.to(device)
            # output = likelihood(model_gp(data))
            output = model_gp(data)
            means = output.mean.detach().cpu().clone()

            #loss = -mll(output, torch.tensor(data.y, dtype=torch.float).to(device).reshape(-1,))
            loss = nn.MSELoss()
            
            test_loss = loss(means, torch.tensor(data.y, dtype=torch.float).reshape(-1,))
            print("mse", test_loss)
            test_loss += test_loss.item()
        test_loss /= len(test_loader.dataset)
    return test_loss, output, data.y


# %%
