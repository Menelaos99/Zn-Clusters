#%%
from training import *
from bo import *
from models import *
import os
import matplotlib.pyplot as plt 
import numpy as np
import warnings
from sklearn.utils import shuffle
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GeoLoader
import gpytorch
from helpers import *
from models import BaseGNN, ExactGPModel
seed = 42
import random
import time
import copy
NR_OF_ABSORBANTS = 1
learning_rate=1e-3
EPOCHS =  1000
SAMPLE_SIZE = 1000
Training_Type = ['Pt3_', 'Pt6_', 'Pt9_', 'Ni3_', 'Ni6_','Pt3Ni3_', 'Ni3Pt2Ni_', 'Pt2Ni_PtNi2_', 'Pt3Pt2Ni_']
Training_Type = ['Pt6_']
Prediction_Type = ['Pt6_']


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
# data_list,data_dict = data_preparation(types_to_include = Training_Type)

# atoms = atom_extraction(Prediction_Type)
# cluster = data_dict[Prediction_Type][0][0][0].pos[:-NR_OF_ABSORBANTS] ##########might be useful 

# cluster_origin = list(zip(atoms[:-NR_OF_ABSORBANTS],cluster))
# print(data_dict)
# #%%
# data_list_9,data_dict_9 = data_preparation('/pt9_cluster',types_to_include = ['Pt9'])
atoms = atom_extraction(Prediction_Type[0])
# atoms = atom_extraction('Pt3_') 
# print(data_dict_9)
# cluster_origin = list(zip(atoms[:],cluster))
# print(data_dict_9)


###

data_list, _ =  data_preparation(types_to_include = Training_Type) ###TODO YOU HAVE TO GIVE THE GRAPHS THAT YOU HAVE CREATED
#data_pred_list, _ =  data_preparation(folder = '/data_prediction/', types_to_include = Prediction_Type) 
print(len(data_list))
#cluster = data_pred_list[0].pos

#cluster_origin = list(zip(atoms[:],cluster))  ### TODO YOU HAVE TO GIVE THE CLUSTER WITHOUT ANY ADSORBANTS. I.E. THE CLUSTER YOU WISH TO OPTIMIZE WITHOUT ZN



#%% cross validation
loss_history = []
eval_loss_history = []
best_model = None
validation_prediction = []
validation_std =[]
validation_truth = []
model_filenames = []
mean_temp =0
std_temp = 0
truth_temp =0
#### TODO here do the cross validation loop as you did in the first project (ignore the data_dict, just put your own loop)

sampling_size = 6
data_len = len(data_list)
num_batches = data_len // sampling_size
indices = np.arange(data_len)
data_list, _ =  data_preparation(types_to_include = Training_Type) 
print(data_list)

#%%
best_vals =[]
best_models = []
best_tests = []
for i in range(num_batches):
    ### Very strange problem keeps happening: On the first cv itearation the extractor and the gp are initialised, but in the second for some reason the data_list changes structure and
    # the code breaks at feature_extractor = BaseGNN(data_list[0].x.shape[-1], data_list[0].edge_attr.shape[-1]) with error 'AttributeError: 'numpy.ndarray' object has no attribute 'x''
    best_val_loss = np.inf

    start_index = i * sampling_size

    val_test_indices = indices[start_index:start_index + sampling_size]
    # np.random.shuffle(val_test_indices)
    
    val_indices = val_test_indices[:3]
    test_indices = val_test_indices[3:]

    train_indices = np.setdiff1d(indices, val_test_indices)
    
    data_list_copy = data_list
    train_data = [data_list_copy[j] for j in train_indices]
    val_data = [data_list_copy[j] for j in val_indices]
    test_data = [data_list_copy[j] for j in test_indices]

    mask = torch.ones(len(data_list))
    mask[val_test_indices] = 0
    mask = mask.bool()
    
    masked_data_list = [data_list[i] for i in range(len(data_list)) if mask[i]]
    
    train_loader = GeoLoader(masked_data_list, batch_size=len(masked_data_list), shuffle=True)
    validation_loader = GeoLoader(val_data, batch_size=len(val_data), shuffle=False)
    test_loader = GeoLoader(test_data, batch_size=len(test_data), shuffle=False)

    noises = torch.ones(len(train_loader)) * 0.1
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noises, learn_additional_noise=True)
    domains = [masked_data_list[i].domain for i in range(len(masked_data_list))]

    # likelihood = gpytorch.likelihoods.GaussianLikelihood()
    ######################### here starts the training of a model per fold#######################
    # print('data list shape',np.array(data_list).shape)
    # print('data list',data_list)

    feature_extractor = BaseGNN(data_list[0].x.shape[-1], data_list[0].edge_attr.shape[-1])
    gp = ExactGPModel(train_x=None, train_y=None, likelihood=likelihood)#, domains=domains
    feature_extractor.to(device)
    gp.to(device)

    targets = []
    
    train_data_copy = np.copy(train_data)
    if_break = None
    for k in range(train_data_copy.shape[0]):
        targets.append(train_data_copy[k][3][1])
    
    targets = torch.tensor(targets, dtype=torch.float).to(device)
    # print(targets)
    model_gp = GNNGP(feature_extractor=feature_extractor, gp=gp, train_x=train_data, train_y=targets)
    model_gp.to(device)

    optimizer = torch.optim.Adam(model_gp.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        factor=0.8, patience=5,
                                                        min_lr=0.0000001)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model_gp)

    best_validation_loss = float('inf')

    for epoch in range(1, EPOCHS+1):
        # print('1', epoch)
        model_gp.train()
        likelihood.train()
        loss, means, std_print = train(train_loader, optimizer, device, model_gp, mll, data_list, domains)
        scheduler.step(loss)
        
        model_gp.eval()
        likelihood.eval()
        # means = model_gp(Batch.from_data_list(train_data)).mean.detach().cpu().clone().numpy()
        # std_print = model_gp(Batch.from_data_list(train_data)).stddev.detach().cpu().clone().numpy()
        validation_loss, validation_output, validation_truth_temp = validate(validation_loader, device, model_gp,likelihood, mll)
        if validation_loss < best_validation_loss:
            test_loss, test_output,test_truth = test(test_loader, device, model_gp,likelihood, mll)
            # print(epoch)
            best_validation_loss = validation_loss
            best_model = model_gp 
            mean_temp = test_output.mean.detach().cpu().clone().numpy()
            std_temp = test_output.stddev.detach().cpu().clone().numpy()
            validation_truth_temp_detached = test_truth.detach().cpu().clone().numpy()

            print('Validation run: %d - Iter %d/%d - Loss: %.3f - Validation Loss: %.3f  \n means : ' % (i,epoch, EPOCHS, loss.item(), validation_loss), means)
            print('Validation run: %d - Iter %d/%d - Loss: %.3f - Validation Loss: %.3f  \n test_means : ' % (i,epoch, EPOCHS, loss.item(), validation_loss), mean_temp)
            print('True_val:',targets.cpu().numpy())
            print('True_val:',test_truth)
            #print(' diff :  ',means-targets.cpu().numpy())#,'\n')
            print(' diff :  ',mean_temp-test_truth.cpu().numpy())#,'\n')
            print('std : ',std_print)
    best_vals.append(best_validation_loss)
    best_tests.append(test_loss)
    best_models.append(best_model)
    validation_prediction.append(mean_temp)    
    validation_truth.append(validation_truth_temp_detached)
    
    eval_loss_history.append({"validation": validation_loss})
model_gp.eval()
likelihood.eval()
##best_val_idx = np.argmin(best_vals)
print(f'total MSE: {sum(best_tests)/len(best_tests)}')

for m in range(len(best_vals)):
    best_model = best_models[m]
    model_filename =  'best_model_' + Prediction_Type[0] + str(m) + '.pth'  
    torch.save(best_model, os.path.join(os.getcwd(),'best_models', model_filename))
    model_filenames.append(model_filename)
# print(model_filename)

loss_history.append({"train": loss})
# validation_prediction.append(mean_temp)
validation_std.append(std_temp)
# validation_truth.append(validation_truth_temp_detached)

#%%
# TODO: plotting 
validation_prediction = array_reshape(validation_prediction)
validation_std = array_reshape(validation_std)
validation_truth = array_reshape(validation_truth)

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'cyan', 'gray', 'gold']
color_map = {}

for i in range(len(validation_truth)//3):
    color_map[i] = colors[i]

for i in range(0, len(validation_truth), 3):
    plt.plot(validation_truth[i:i+3], validation_prediction[i:i+3], 'o', color=color_map[i//3])
plt.plot([min(validation_truth), max(validation_truth)], [min(validation_truth), max(validation_truth)], color='red', linestyle='--', label='y = x')
legend_labels = [f"Triplet {i+1}" for i in range(len(validation_truth)//3)]
plt.legend(legend_labels, loc='upper right')
# plt.plot(validation_truth,validation_prediction, 'o')
plt.xlabel('Actual Labels')
plt.ylabel('Predictions')
# plt.xlim(-1.5,2)
# plt.ylim(-1.5,2)

#%%
#################### 
opt = 0
avg_state_dict = {}
for filename in model_filenames:
    model = torch.load(os.path.join(os.getcwd(),'best_models', filename))
    opt += optimum_energy(model, data_list) 
print(opt/len(model_filenames))
opt = - np.inf ## if the cluster is unknown 

# %%
# avg_dict = {}
# for filename in model_filenames:
#     model_state_dict = torch.load(os.path.join(os.getcwd(), filename))

#     for key,_ in model_state_dict.items():
#         if key not
start_time = time.time()                                 
#model_gp = torch.load(os.path.join(os.getcwd(), filename))
#opt = optimum_energy(model_gp, data_list)
bo = BO(cluster,atoms,opt, likelihood, model_filenames, method = "pe", tradeoff = 0.5, sample_size = SAMPLE_SIZE ,device = device)
bo.create_predictions()
print("--- %s seconds ---" % (time.time() - start_time))
#%%

name = 'Pt2NiPt2Ni_BO_1,to_1'
nr = 1
bo.tradeoff = 1
bo.method = 'pe'
save_points(name, bo.get_score()[1],cluster_origin, nr)
bo.tradeoff = 0
# name = 'exploration_PT9'
# nr = 2
# save_points(name, bo.get_score()[1],cluster_origin, nr)
# # bo.method = 'ucb'

# print(bo.get_score()[1])

#%%


# plt.scatter(bo.Xsamples[:].pos[:,0], bo.Xsamples[:].pos[:,1],bo.Xsamples[:].pos[:,2], s=200, c=bo.std, cmap='gray')
# %%
positions = []
for i in range(len(bo.Xsamples)):
    positions.append(bo.Xsamples[i].pos[-1,:])

positions = np.array(positions)

#%%


fig = plt.figure(figsize=(10, 10))

axs = fig.add_subplot(projection='3d')
p =  axs.scatter(positions[:,0], positions[:,1], positions[:,2],c = bo.mu[:],s = 50)
fig.colorbar(p, ax=axs)
# axs.scatter(-2.846607  , 17.652020 , 10.442366, s = 200)
# axs.scatter( 2.144176 ,12.477394, 13.453673, s = 200)

# %%
bo.std[:]
axs.scatter(cluster[:,0],cluster[:,1],cluster[:,2], s = 200)
# fig.colorbar(p, ax=axs)

# %%


cluster_pos = bo.Xsamples[0].pos[:-1,:]
fig = plt.figure(figsize=(10, 10))

axs = fig.add_subplot(projection='3d')
p =  axs.scatter(cluster_pos[:,0], cluster_pos[:,1], cluster_pos[:,2],s = 50)
bo.tradeoff = 0
bo.method = 'pe'
p =  axs.scatter(bo.get_score()[1][:,0], bo.get_score()[1][:,1], bo.get_score()[1][:,2],s = 50 )

name = "Pt2NiPt2Ni_BO_pe_0"
#save_points(name, bo.get_score()[1],cluster_origin, nr)

# %%
