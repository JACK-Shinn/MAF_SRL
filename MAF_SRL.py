'''
   Code author: Xin Xu
   Email: 2208067587@qq.com

 '''

import torch
from torch import nn
import copy
from tqdm import tqdm
import numpy as np
from scipy.sparse import random as sp_random
from sklearn.preprocessing import normalize
from clusteringPerformance import StatisticClustering, similarity_function

class MAF_SRL(nn.Module):
    def __init__(self, block_num, epoch, lr, W, nc, gnd, 
                 sparsity=0.3, init_scale=0.1):
        super(MAF_SRL, self).__init__()
        self.block_num = block_num
        self.epoch = epoch
        self.lr = lr
        self.nc = nc
        self.gnd = gnd
        
        # Initialize input data
        self.W = torch.from_numpy(W).float()
        self.n_nodes = W.shape[0]
        
        # Multivariate activation parameters
        self.sparsity = sparsity
        self.init_scale = init_scale
        
        # Initialize parameter matrices
        self.A = self._init_sparse_matrix(self.n_nodes, sparsity, init_scale)
        self.q = nn.Parameter(torch.randn(self.n_nodes))  # Feature scaling parameters
        self.b = nn.Parameter(torch.zeros(self.n_nodes))  # Offset parameters
        
        # Activation function parameters
        self.eta1 = nn.Parameter(torch.tensor(1.0))
        self.eta2 = nn.Parameter(torch.tensor(0.5))
        self.delta1 = nn.Parameter(torch.tensor(0.5))
        self.delta2 = nn.Parameter(torch.tensor(1.0))
        
        # Lipschitz constant
        self.L = nn.Parameter(torch.tensor(1.0))

    def _init_sparse_matrix(self, dim, sparsity, scale):
        """Initialize sparse transformation matrix"""
        sparse_mat = sp_random(dim, dim, density=sparsity, 
                             data_rvs=lambda x: scale*np.random.randn(x))
        indices = torch.LongTensor(np.vstack((sparse_mat.row, sparse_mat.col)))
        values = torch.FloatTensor(sparse_mat.data)
        return torch.sparse_coo_tensor(indices, values, (dim, dim))

    def _project_parameters(self):
        """Project parameters to ensure constraints"""
        with torch.no_grad():
            # Ensure q remains positive
            self.q.data = torch.clamp(self.q, min=1e-4)
            # Keep eta2 in (0,1] range
            self.eta2.data = torch.clamp(self.eta2, 1e-4, 1.0)
            # Maintain delta ordering
            self.delta1.data = torch.clamp(self.delta1, min=0)
            self.delta2.data = torch.clamp(self.delta2, min=self.delta1.item())

    def _multivariate_activation(self, x):
        """Core of multivariate activation function"""
        # Convert to dense matrix for computation
        A_dense = self.A.to_dense()
        q_diag = torch.diag(self.q)
        
        # Step 1: Linear transformation
        lin_transform = torch.linalg.inv(A_dense @ q_diag) @ x
        
        # Step 2: Apply univariate activation
        activated = self._univariate_activation(lin_transform)
        
        # Step 3: Inverse transformation
        return torch.linalg.inv(A_dense.T) @ (activated - self.b)

    def _univariate_activation(self, x):
        """Improved piecewise linear activation function"""
        mask1 = x >= self.delta2
        mask2 = (x >= self.delta1) & (x < self.delta2)
        mask3 = (x >= -self.delta1) & (x < self.delta1)
        mask4 = (x >= -self.delta2) & (x < -self.delta1)
        mask5 = x < -self.delta2
        
        output = torch.zeros_like(x)
        output[mask1] = self.eta2*(x[mask1]-self.delta2) + self.eta1*(self.delta2-self.delta1)
        output[mask2] = self.eta1*(x[mask2]-self.delta1)
        output[mask4] = self.eta1*(x[mask4]+self.delta1)
        output[mask5] = self.eta2*(x[mask5]+self.delta2) + self.eta1*(self.delta1-self.delta2)
        return output

    def forward(self, W):
        """Forward propagation process"""
        W_list = [W]
        for _ in range(self.block_num):
            # Residual connection
            residual = W_list[-1] - (1/self.L) * (W_list[-1] - W)
            # Multivariate activation
            activated = self._multivariate_activation(residual)
            W_list.append(activated)
        return W_list

    def my_loss1(self, x, pred_x):
        """Improved reconstruction loss function"""
        return 0.5 * torch.norm(x - pred_x)**2 + 1e-6*torch.norm(self.A, p=1)

    def train(self):
        """Training process"""
        self.loss_list = []
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr,
            betas=(0.90, 0.92),
            weight_decay=0.15
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            factor=0.1, 
            patience=3, 
            verbose=True, 
            min_lr=1e-6
        )
        
        with tqdm(total=self.epoch, desc="Training") as pbar:
            for epoch_id in range(self.epoch):
                optimizer.zero_grad()
                
                # Forward propagation
                W_list = self(self.W)
                loss = self.my_loss1(self.W, W_list[-1])
                
                # Backward propagation
                loss.backward()
                optimizer.step()
                self._project_parameters()  # Parameter projection
                
                # Record loss
                train_loss = loss.detach().item()
                self.loss_list.append(train_loss)
                
                # Learning rate adjustment
                scheduler.step(loss)
                
                # Early stopping mechanism
                if optimizer.param_groups[0]['lr'] <= 2e-7:
                    print("Early stopping triggered")
                    break
                
                pbar.update(1)

    def get_ans(self):
        """Retrieve final reconstructed matrix"""
        with torch.no_grad():
            W_list = self(self.W)
            return W_list[-1].cpu().numpy()

    def spectral_clustering(self, points, k, repnum=10):
        """Spectral clustering process"""
        W = similarity_function(points)
        Dn = np.diag(1 / np.power(np.sum(W, axis=1), -0.5))
        L = np.eye(len(points)) - np.dot(np.dot(Dn, W), Dn)
        
        eigvals, eigvecs = np.linalg.eig(L)
        eigvecs = eigvecs.astype(float)
        indices = np.argsort(eigvals)[:k]
        k_smallest_eigenvectors = normalize(eigvecs[:, indices])
        
        ACC, NMI, ARI = StatisticClustering(
            k_smallest_eigenvectors, 
            self.gnd, 
            k, 
            repnum
        )
        print(f"Clustering Metrics - ACC: {ACC:.4f}, NMI: {NMI:.4f}, ARI: {ARI:.4f}")
        return [ACC, NMI, ARI]