#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:47:14 2023

@author: rileywilde
"""


import os
import argparse
import torch

import numpy as np
import graphlearning as gl
from scipy import sparse
import scipy.sparse as sps
from scipy.special import softmax
import os
import sys
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm

from networks.resnet_big import SupConResNet, SupConResNetHead
import torch
import torchvision.transforms as T
import util as utils
import matplotlib.pyplot as plt
from active_learning import *

import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE
import scipy.sparse as sps
from scipy import sparse
import pandas as pd
import graphlearning as gl
import torch
import numpy as np
import sys
#sys.path.append('/content/drive/MyDrive/Sar-Ship-2.0/MSTAR-Active-Learning/Python')
from active_learning import *

def umap_visualization(encoded_data, labels, data_name = '', n_neighbors=15, sel_labels=None):
  reducer = umap.UMAP(n_neighbors = n_neighbors)
  umap_embedded_data = reducer.fit_transform(encoded_data)

  plt.figure()
  plt.scatter(umap_embedded_data[:,0], umap_embedded_data[:,1], c=labels, s=.5)
  if sel_labels is not None:
    plt.scatter(umap_embedded_data[sel_labels,0], umap_embedded_data[sel_labels,1], c='r', s=1, marker='*')
  plt.title("UMAP Embedding of " + data_name +" Data")
  plt.show()
  #plt.savefig(os.path.join("..", "results", f"umap_{args.vae_fname}.png"))
def tsne_visualization(encoded_data, labels, data_name = '', sel_labels=None):
  tsne_embedded_data = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(encoded_data)
  plt.figure()
  plt.scatter(tsne_embedded_data[:,0], tsne_embedded_data[:,1], c=labels, s=.5)
  if sel_labels is not None:
    plt.scatter(tsne_embedded_data[sel_labels,0], tsne_embedded_data[sel_labels,1], c='r', s=1, marker='*')
  plt.title("TSNE Embedding of " + data_name +" Data")
  plt.show()
  


def create_graph(encoded_data, num_neighbors=50):
  X = encoded_data
  knn_data = gl.weightmatrix.knnsearch(X,num_neighbors,similarity='angular')   
  #Build weight matrix
  W = gl.weightmatrix.knn(None,num_neighbors,knn_data=knn_data)
  N = W.shape[0]
  # Calculate eigenvalues and eigenvectors of unnormalized graph Laplacian if not previously calculated
  print("Calculating Eigenvalues/Eigenvectors...")
  L = sps.csgraph.laplacian(W, normed=False)
  M=200
  evals, evecs = sparse.linalg.eigsh(L, k=M+1, which='SM')
  evals, evecs = evals.real, evecs.real
  evals, evecs = evals[1:], evecs[:,1:]  # we will ignore the first eigenvalue/vector
  # Also compute normalized graph laplacian eigenvectors for use in some GraphLearning graph_ssl functions (e.g. "mbo")
  n = W.shape[0]
  G = gl.graph(W)
  deg = G.degree_vector()
  m = np.sum(deg)/2
  gamma = 0
  Lnorm = G.laplacian(normalization="normalized")
  
  def Mnorm(v):
      v = v.flatten()
      return (Lnorm*v).flatten() + (gamma/m)*(deg.T@v)*deg
  
  Anorm = sparse.linalg.LinearOperator((n,n), matvec=Mnorm)
  vals_norm, vecs_norm = sparse.linalg.eigs(Anorm,k=300,which='SM')
  vals_norm = vals_norm.real; vecs_norm = vecs_norm.real
  return W, evals, evecs, vals_norm, vecs_norm


def active_learning_loops(encoded_data, labels, num_neighbors=50, train_idx_all=None, test_mask=None, umap=False, num_iters=290, num_trials = 1, seed=2, num_per_class=1):
  #Create the graph 
  W, evals, evecs, vals_norm, vecs_norm = create_graph(encoded_data, num_neighbors)
  METHODS = ['random', 'uncertainty', 'mc', 'mcvopt', 'vopt']
  gamma = 0.5
  results_df = pd.DataFrame([])
  if train_idx_all is None:
    train_idx_all = np.arange(encoded_data.shape[0])

  for acq in METHODS:
      print(f"Acquisition Function = {acq.upper()}") 
      # Run Active Learning Test for this current acqusition function
      accuracy = np.zeros(num_iters+1)
      trials = num_trials
      for i in np.arange(trials):
          train_ind = np.array([], dtype=np.int16)
          for c in np.sort(np.unique(labels)):
              c_ind = np.intersect1d(np.where(labels == c)[0], train_idx_all) # ensure the chosen points are in the correct subset of the dataset
              if trials == 1:
                  rng = np.random.default_rng(seed) # for reproducibility
                  train_ind = np.append(train_ind, rng.choice(c_ind, num_per_class, replace=False))
              else:
                  train_ind = np.append(train_ind, np.random.choice(c_ind, num_per_class, replace=False))
              
          train_ind, accuracy_temp = active_learning_loop(W, evals, evecs, train_ind, labels, num_iters, acq, train_idx_all=train_idx_all, \
                                      test_mask=test_mask, gamma=gamma, by_class=False, algorithm="laplace", vals_norm=vals_norm, vecs_norm=vecs_norm)
          accuracy += accuracy_temp

      results_df[acq+"_choices"] = np.concatenate(([-1], train_ind[-num_iters:]))
      results_df[acq+"_acc"] = accuracy/trials

      print("\n")
  

  x = np.arange(num_per_class, num_per_class + num_iters + 1)

  # General plot settings
  legend_fontsize = 12
  label_fontsize = 16
  fontsize = 16
  # matplotlib.rcParams.update({'font.size': fontsize})
  #styles = ['^b-','or-','dg-','pm-','xc-','sk-', '*y-']

  skip = 5
  '''
  plt.figure()
  for i, method in enumerate(METHODS):
      plt.plot(x[::skip], 100*results_df[method + "_acc"][::skip], styles[i], label = method + " accuracy")

  plt.xlabel("Number of Labeled Points")
  plt.ylabel("Accuracy %")
  plt.title(f"Active Learning with Contrastive Representations")
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.show()
  '''
  return x,100*results_df
  

def encode_data(model_path, data, remove_parallel=True):
  model = SupConResNetHead(name="resnet18")
  stuff = torch.load(model_path)
  if remove_parallel:
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in stuff["model"].items():
        name = k.replace(".module", "") # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

  else:
    model.load_state_dict(stuff["model"])

  device = torch.device("cuda")
  model.to(device)

  batch_size = 1000
  stuff = torch.load(model_path)
  model.load_state_dict(stuff["model"])
  model.eval()
  encoded_data = None
  with torch.no_grad():
    for idx in range(0,len(data),batch_size):
        data_batch = data[idx:idx+batch_size]
        if encoded_data is None:
            encoded_data = model.encoder(data_batch.to(device)).cpu().numpy()
        else:
            encoded_data = np.vstack((encoded_data,model.encoder(data_batch.to(device)).cpu().numpy()))

  return encoded_data


path2mstar_models = './mstar_models' #don't want tensorboard 
save_dir = './save/model_comp'
if os.path.exists(save_dir)==False:
    os.mkdir(save_dir)
    

PATHS = []
for path, subdirs, files in os.walk(path2mstar_models):
    for name in files:
        PATHS.append(os.path.join(path, name))
        

#def main():
for q in [1]:
    ## load data:
    hdr, fields, mag, phase = utils.load_MSTAR()
    # Get labels and corresponding target names
    train_mask, test_mask, _ = utils.train_test_split(hdr,1)
    labels, target_names = utils.targets_to_labels(hdr)
    

    use_phase = True
    if use_phase:
        data = utils.polar_transform(mag,phase)
    else:
        data = np.reshape(mag,(mag.shape[0],1,mag.shape[1],mag.shape[2]))

    data = torch.from_numpy(data).float()

    resize_t = T.Resize(32)
    data = resize_t(data)
    
    acc500 = []
    names500 = []
    acc1000 = []
    names1000 = []
    
    i=0
    for fpath in PATHS: 
        i=i+1
        fname = ''
        for c in fpath[len(path2mstar_models)+1:]:
            if c !='/':
                fname=fname+c
        fname = fname[:-4] #remove .pth
        
        if fname[-4:] =='last':
            continue #redundant 
        
        enc = encode_data(fpath,data)
        print(fname)
        
        x,acc_dfs = active_learning_loops(enc, labels, train_idx_all=np.where(train_mask)[0])
        
        
        keys = ['random_acc', 'uncertainty_acc', 'mc_acc', 'mcvopt_acc', 'vopt_acc']
        
        
        if fname[-3:]=='500':
            acc500.append(np.vstack([acc_dfs[k] for k in keys]))
            names500.append(fname)
        else: #last or 1000
            acc1000.append(np.vstack([acc_dfs[k] for k in keys]))
            names1000.append(fname)
            
    
    a500 = np.array(acc500)
    a1000 = np.array(acc1000)
    #dim 1 
    
    n500 = np.array(names500)
    n1000 = np.array(names1000)
    
    import pickle
    
    with open('./active_learning_data.pckl', 'wb') as f:
        pickle.dump([keys,n500,a500,n1000,a1000],f)
    
    with open('./active_learning_data2.pckl', 'wb') as f:
        pickle.dump([keys,n500,a500,n1000,a1000],f)
    #I'm saving twice just in case I accidentally overwrite one with wb instead of rb lol
    
    
    
    