# Copyright (C) 2012 Anaconda, Inc
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import ot
#import mat
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()  
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
from scipy.stats import chi

##################################################################################################################################
# ProgOT 
from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.tools import progot
import jax.numpy as jnp
from typing import Any, Optional

def run_progot(
    x: jnp.ndarray, y: jnp.ndarray, cost_fn: costs.TICost, **kwargs: Any
) -> progot.ProgOTOutput:
    geom = pointcloud.PointCloud(x, y, cost_fn=cost_fn)
    prob = linear_problem.LinearProblem(geom)
    estim = progot.ProgOT(**kwargs)
    out = estim(prob)
    return out

def MK_depth(data):
    ''' data in pca space. Returns vector-valued ranks in the unit ball, MK depth, and the solution of ProgOT. 
    '''
    d = data.shape[1]
    Ud = sample_reference(10*data.shape[0],data.shape[1]) # np.random.random(data.shape) 
    epsilons = jnp.array([1,0.5,0.1,0.01])
    K = len(epsilons)
    alphas = progot.get_alpha_schedule("exp", num_steps=K)
    cost_fn = costs.SqEuclidean()
    df_x, df_y = jnp.array(data) ,jnp.array(Ud) 
    out = run_progot(df_x, df_y , cost_fn, alphas=alphas, epsilons=epsilons)
    xs, ranksMK = out.transport(df_x, return_intermediate=False)
    depthMK = HalfspaceDepth_pts(ranksMK) #1/ (1+np.linalg.norm(ranksMK,axis=1))
    return(ranksMK,depthMK,out,Ud)

def HalfspaceDepth_pts(df):
    ''' Halfspace depth on the spherical uniform distributon Ud. df has size (n,d) for n points in dimension d'''
    tau = np.linalg.norm(df,axis=1) 
    tht = np.arccos(np.minimum(1, tau))
    HD_u = (tht - np.cos(tht)*np.log(np.abs( (1 + np.sin(tht))/ np.cos(tht)  )) )/np.pi
    return(HD_u)

def HalfspaceDepth(u):
    ''' u is a single point. Halfspace depth on spherical distribution Ud, from CHERNOZHUKOV, GALICHON, HALLIN, HENRY. '''
    tau = np.linalg.norm(u)
    tht = np.arccos(tau)
    HD_u = (tht - np.cos(tht)*np.log(np.abs( (1 + np.sin(tht))/ np.cos(tht)  )) )/np.pi
    return(HD_u)

def PCA_MK_Depth(V,n_components=0.95):
    ''' df : data-frame of images, of size (n,p,p) for n images with p*p pixels. 
    This function embeds data using LOT, performs PCA, and learns MK quantiles. 
    ''' 
    # PCA 
    pca = decomposition.PCA(n_components=n_components, svd_solver="auto") 
    data = pca.fit_transform(V) 
    #print("PCA space dimension:",data.shape)
    #if dim==None: 
    #    dim = pca.n_components_ 
    # Decomposition Signal / Noise 
    #data_signal =  data[:,:dim] 
    #data_noise = data[:,dim:] 
    #eigenvalues = pca.explained_variance_ 
    #var_noise = np.var(data_noise) ## Alternatively : np.mean(eigenvalues[dim:]) 
    # MK ranks for the noise 
    #normnoise = np.linalg.norm(data_noise,axis=1).reshape(-1,1)
    #ranksMK_noise = (data_noise/ normnoise) * chi.cdf( normnoise,df=data_noise.shape[1],scale=np.sqrt(var_noise) )
    #depthMK_noise = HalfspaceDepth_pts(ranksMK_noise) #1/(1+np.linalg.norm(ranksMK_noise,axis=1))
    ranksMK_signal,inner_depth,out,Ud = MK_depth(data)
    Back_projection = pca.inverse_transform(data)
    outer_depth =  1/(1+np.linalg.norm(V - Back_projection,axis=1)) 
    ########### 
    return(out,pca,inner_depth,data,outer_depth)

def ProgOT(data):
    ''' data in pca space. Learns the MK quantile function, from a reference distribution to the data.
    '''
    Ud =  sample_reference(data.shape[0],data.shape[1]) # np.random.random(data.shape)
    epsilons = jnp.array([1,0.5,0.1,0.01,0.005,0.001,0.0001])
    Kk = len(epsilons)
    alphas = progot.get_alpha_schedule("exp", num_steps=Kk)
    cost_fn = costs.SqEuclidean()
    df_x, df_y =  jnp.array(Ud) ,jnp.array(data)
    out = run_progot(df_x, df_y , cost_fn, alphas=alphas, epsilons=epsilons)
    xs, Qx = out.transport(df_x, return_intermediate=False)
    return(out,Qx,Ud)


###############################################################################################
# LOT EMBEDDING
###############################################################################################

def grid_pixels(p1,p2):
    x = np.linspace(0,p1-1,p1)
    y = np.linspace(0,p2-1,p2)
    grid_pixels = []
    for i in range(0,p2)[::-1]:
        for j in range(0,p1):
            grid_pixels.append([x[j]/p1, y[i]/p2])
    return( np.array(grid_pixels) )

def sample_from_pdf(grid_points,nu,n_sample = 500):
    nu = nu/np.sum(nu)
    idx = np.random.choice(np.arange(grid_points.shape[0]), p=nu,size=n_sample) 
    return(grid_points[idx])

def LOT_embed1(img,reference_data):
    ''' Embed one image 'img' that must be of size (p,p) for p*p pixels '''
    p1,p2 = img.shape
    n_sample = len(reference_data)
    img = img.flatten() / np.sum(img)
    target_data = sample_from_pdf(grid_pixels(p1,p2),img,n_sample=n_sample)
    C = ot.dist(reference_data,target_data)
    # OT Problem 
    alpha = np.ones(n_sample)/n_sample
    M = ot.emd(alpha,alpha,C)*n_sample
    # Transport map from barycentric projection of the OT plan
    T = np.matmul(M,target_data) # image by Monge Map of reference data, this shall recover the input 'img'. 
    V = T - reference_data # vector map 
    return(V)

def LOT_transform1(df,n_sample=501):
    ''' To embed a dataframe 'df' in LOT space. The reference is taken to the mean image of df. `n_sample` determines the number of dirac positions for the reference measure. '''
    # Reference point / mother image 
    central = np.mean(df,axis=0)
    p1,p2 = df.shape[1],df.shape[2] 
    L2dist =  np.linalg.norm(   (df - central).reshape(len(df), p1*p2 )   ,axis=1) 
    reference_img = df[np.argmin(L2dist)] 
    reference_img = reference_img.reshape(p1*p2) / np.sum(reference_img)
    reference_data = sample_from_pdf(grid_pixels(p1,p2),reference_img,n_sample)
    # LOT  
    V = [] 
    for X in df: 
        Vi = LOT_embed1(X,reference_data) 
        V.append(np.ravel(Vi))
    V = np.array(V)
    #print("LOT space shape (n,d):",V.shape)
    return(V,reference_data)

#def LOT_transform_bar(df,n_sample=501):
#    ''' To embed a dataframe 'df' in LOT space. The reference is learnt as an approximate Wasserstein barycenter. '''
#    p = df.shape[1]
#    # Learning a central image
#    reference_img = np.mean(df,axis=0)
#    reference_img = reference_img.reshape(p*p) / np.sum(reference_img) 
#    reference_data = sample_from_pdf(grid_pixels(p),reference_img,n_sample)
#    V = [] 
#    for X in df: 
#        Vi = LOT_embed(X,reference_data) 
#        V.append(np.ravel(Vi))
#    V = np.array(V)
#    pca = decomposition.PCA(n_components=10, svd_solver="auto")
#    V = pca.fit_transform(V)
#    reference_img = df[np.argmin( np.linalg.norm(V,axis=1) )]
#    # Embedding wrt Tangent plan 
#    reference_img = reference_img.reshape(p*p) / np.sum(reference_img) 
#    reference_data = sample_from_pdf(grid_pixels(p),reference_img,n_sample)
#    V = [] 
#    for X in df: 
#        Vi = LOT_embed(X,reference_data)
#        V.append(np.ravel(Vi))
#    V = np.array(V)
#    return(V,reference_data)

def LOT_embed_inverse(point,reference_data):
    point = point.reshape(reference_data.shape)
    point = (point + reference_data)
    return(point)

def From_Reduced_LOT_to_Img(point,scaler,pca,reference_data):
    ''' Send a point from LOT space (i.e. reduced by ACP and scaled between the unit ball) to Image space'''
    # From the reduced LOT space to the LOT space 
    point = scaler.inverse_transform(point.reshape(1, -1)) 
    point = pca.inverse_transform(point)
    # to go back to image space :
    point = LOT_embed_inverse(point,reference_data)
    return(point)

from scipy.stats import gaussian_kde
def pointcloud2image(point):
    ''' Kernel smoothing to render a density image, given a point cloud in [0,1]^2. '''
    xmin,xmax,ymin,ymax = 0,1,0,1
    NB = 100j
    X, Y = np.mgrid[xmin:xmax:NB, ymin:ymax:NB]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = point.T
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    return(np.rot90(Z))

def mahalanobis(x, pca):
    m = pca.mean_
    cov = pca.get_covariance()
    x_mu = x -  pca.mean_ 
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left, x_mu.T)
    return mahal 

def MahaDepth(points,pca):
    ''' points must contain several observations in the LOT space, where pca has been computed. '''
    mahal = mahalanobis(points, pca)
    return( 1 / ( 1+mahal.diagonal() ) )

##################################################################################################################################
### PYTRANSKIT 
##################################################################################################################################

#from pytranskit.optrans.utils import data_utils
#from pytranskit.optrans.continuous.clot import CLOT

#def LOT_embed(img,reference_img,sigma=1.,lr=0.):
#    ''' Embed one image '''
#    p1,p2 = img.shape
#    img = data_utils.signal_to_pdf(img, sigma=sigma, total=1.)
#    clot = CLOT(max_iter=500, lr=lr, tol=1e-1,verbose=0.)
#    # calculate CLOT
#    lot = clot.forward(img, reference_img)
#    ## transport map and displacement map from reference_img to img
##    tmap10 = clot.transport_map_ 
#    # apply forward map to transport I1 (reference_img) to I0 (img)
#    #img0_recon = clot.apply_forward_map(tmap10, reference_img)

#    V = tmap10.T - grid_pixels(p1,p2).reshape((p1,p2,2)) 
#    return(V)

#def LOT_transform(df,sigma=1.,lr=0.):
#    ''' To embed a dataframe 'df' in LOT space. The reference is taken to the mean image of df.  '''
#   # Reference point / mother image 
#    central = np.mean(df,axis=0)
#    p1,p2 = df.shape[1],df.shape[2] 
#    L2dist =  np.linalg.norm(   (df - central).reshape(len(df), p1*p2 )   ,axis=1) 
#    reference_img = df[np.argmin(L2dist)] 
#   
#    reference_img = data_utils.signal_to_pdf(reference_img, sigma=sigma, total=1.)
#    # LOT  
#    V = [] 
#    for X in df: 
#        Vi = LOT_embed(X,reference_img,sigma=sigma,lr=lr) 
#        V.append(Vi.flatten())
#    V = np.array(V)
#    #print("LOT space shape (n,d):",V.shape)
#    return(V,reference_img)


##################################################################################################################################
# Diverse 
##################################################################################################################################
    
def cost(x,Y):
	''' Squared euclidean distance between x and the J points of Y.
	'''
	diff = Y-x 
	return(0.5*(np.linalg.norm(diff,axis=1))**2)

def sample_reference(N,d):
    Ud = np.zeros((N,d))
    Nn = int(N/2)
    radius = np.linspace(0,0.9,Nn+1)[1:]
    compteur = 0 
    for i in range(Nn):
        dir = np.random.normal(size=d)
        r = radius[i]
        Ud[compteur] = r* dir/np.linalg.norm(dir)
        Ud[compteur+1] = -r* dir/np.linalg.norm(dir)
        compteur = compteur + 2
    if (N-2*Nn >0):
        Ud[compteur] = np.zeros(d)
    return(Ud)


    
    








 