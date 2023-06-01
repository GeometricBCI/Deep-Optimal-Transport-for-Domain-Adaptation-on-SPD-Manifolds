'''
#####################################################################################################################
Date        : 1st, Jun., 2023
---------------------------------------------------------------------------------------------------------------------
Descriptions:  This is the part of the code for visualization, where you can observe the source-target distribution 
and the distribution of the source after it has been moved through optimal transport.

The source code for SPDSW can be found in the following link:

https://github.com/clbonet/SPDSW

#######################################################################################################################
'''


from vedo import *
import ot
import geoopt
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange
from geoopt import linalg
from pyriemann.datasets import sample_gaussian_spd, generate_random_spd_matrix
from EMD_OT import EMD_SPD


import torch as th
import torch.nn.functional as F
import torch.distributions as D
device = "cuda" if th.cuda.is_available() else "cpu"


def busemann_spd(logM, diagA):
    C = diagA[None] * logM[:,None]
    return -C.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

def proj_geod(theta, M):
    """
        - theta in S^{d-1}
        - M: batch_size x d x d: SPD matrices
    """
    A     = theta[:,None] * th.eye(theta.shape[-1], device=device)
    
    ## Preprocessing to compute the matrix product using a simple product
    diagA = th.diagonal(A, dim1=-2, dim2=-1)
    dA    = diagA.unsqueeze(-1)
    dA    = dA.repeat(1,1,2)    
    n_proj, d, _ = dA.shape
    log_M = linalg.sym_logm(M)
    Mp    = busemann_spd(log_M, dA).reshape(n_proj, -1)
    
    return th.exp(-Mp[:,:,None] * diagA[:,None]), Mp

def mat2point(tab_mat):
    return np.concatenate([[tab_mat[:,0,0]], [tab_mat[:,1,1]], [tab_mat[:,0,1]]], axis=0).T

mean0     = np.eye(2)
sigma0    = 0.4
n_samples = 50


W       = th.tensor([[[1, 0.5],[0.5, 1]]], dtype=th.float64)
B0      = th.tensor(sample_gaussian_spd(n_matrices=n_samples, mean=mean0, sigma=sigma0), device=device)
Wt_B0_W = th.matmul(th.matmul(th.transpose(W, 2, 1), B0), W)
vect_B0 = mat2point(B0.numpy())
vect_Wt_B0_W = mat2point(Wt_B0_W.numpy())


theta_0 = np.random.normal(size=(1, 2))
theta_0 = F.normalize(th.from_numpy(theta_0), p=2, dim=-1).to(device)
proj_B0, buseman_coord_0 = proj_geod(theta_0, B0)


B1           = B0
Wt_B1_W      = EMD_SPD(B0, Wt_B0_W, metric="le")
vect_B1      = mat2point(B1.numpy())
vect_Wt_B1_W = mat2point(Wt_B1_W.numpy())


theta_1 = np.random.normal(size=(1, 2))
theta_1 = F.normalize(th.from_numpy(theta_1), p=2, dim=-1).to(device)
proj_B1, buseman_coord_1 = proj_geod(theta_1, B1)


#%% DISPLAY PART

# create cone
height       = 5
res          = 10
light        = "off" 
cone_mesh1   = Plane(pos=(height/2, height/2, 0), normal=(0, 0, 1), s=(height, height), alpha=.2, res=(res, res))
cone_points1 = cone_mesh1.points()
cone_mesh2   = Plane(pos=(height/2, height/2, 0), normal=(0, 0, 1), s=(height, height), alpha=.2, res=(res, res))
cone_points2 = cone_mesh2.points()


for i in range(cone_points1.shape[0]):
    cone_points1[i,2] =  np.sqrt(cone_points1[i,0]*cone_points1[i,1])
    cone_points2[i,2] = -np.sqrt(cone_points2[i,0]*cone_points2[i,1])

cone_mesh1.points(cone_points1).compute_normals().lighting(light)
cone_mesh2.points(cone_points2).compute_normals().lighting(light)


ts = th.linspace(0, 1, 100)
tab_proj_0 = []
tab_proj_1 = []


for i in range(len(vect_B0)):

    proj_B0_diag = proj_B0[0,i][:,None] * th.eye(2)
    geod_le_0    = linalg.sym_expm((1-ts)[:,None,None] * linalg.sym_logm(B0[i]) + ts[:,None,None] * linalg.sym_logm(Wt_B0_W[i]))
    tab_proj_0.append(Line(mat2point(geod_le_0.numpy())).color('blue').lw(0.8))


for i in range(len(vect_B1)):

    proj_B1_diag = proj_B1[0,i][:,None] * th.eye(2)
    geod_le_1    = linalg.sym_expm((1-ts)[:,None,None] * linalg.sym_logm(B1[i]) + ts[:,None,None] * linalg.sym_logm(Wt_B1_W[i]))
    tab_proj_1.append(Line(mat2point(geod_le_1.numpy())).color('blue').lw(0.8))


s_0   = Spheres(vect_B0, r=.1).c("red")
wsw_0 = Spheres(vect_Wt_B0_W, r=.1).c("blue")
s_1   = Spheres(vect_B1, r=.1).c("red")
wsw_1 = Spheres(vect_Wt_B1_W, r=.1).c("blue")


plt   = Plotter(N=2, bg='white', axes=0)


vp    = plt.at(0).show(cone_mesh1,cone_mesh2, s_0, wsw_0, "Source-Target", zoom=1.2, interactive=0)
vp    = plt.at(1).show(cone_mesh1,cone_mesh2, tab_proj_1, s_1, wsw_1, "Source - Source (OT)", zoom=1.2, interactive=1)
vp.screenshot('Source_OT.png')





