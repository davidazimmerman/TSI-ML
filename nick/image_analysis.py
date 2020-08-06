#%%
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:28:13 2019

@author: nrb171
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import scipy as sp
import scipy.interpolate as spinterp
import time
import pickle
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from scipy.ndimage.measurements import label as bwlabel

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
#from numpy.ndarray import flatten

#%%

indiv_frames = '/data/lasso/sims/20160611/indiv_frames/'
files = os.listdir(indiv_frames)
# split up the files into masked and rendered

# D: Why are we masking files? Are some files training examples
# and others truth values? 
mask = np.zeros(len(files))
for i in range(len(files)):
    if 'MASKED' in files[i]:
        mask[i] = 2
    else:
        mask[i] = 1
   
files = np.array(files)     
files_mask = files[mask == 2]
files_rend = files[mask == 1]

#%%
###TODO


number_files_to_loop = 500
#img = cv2.imread(indiv_frames+files[1])

#load an image in
# D: Why are we reducing size? 
resize_ratio = 0.5
sizex = round(480*resize_ratio)
sizey = round(640*resize_ratio)

# D: need descriptions of each of these arrays
# gradient along inside of entire cloudy object
whole_gradient = np.zeros((sizey, sizex, number_files_to_loop))
# gradient from pixel to 10 pixels in front
front_gradient = np.zeros((sizey, sizex, number_files_to_loop))
# gradient from pixel to 10 pixels behind
rear_gradient = np.zeros((sizey, sizex, number_files_to_loop))
# gradient from 10 pixels out on either side
middle_gradient = np.zeros((sizey, sizex, number_files_to_loop))
# average brightness
average_near_pixel = np.zeros((sizey, sizex, number_files_to_loop))
# labeled image 
label = np.zeros((sizey, sizex, number_files_to_loop ))
# brightness of image (sum of 2-norms)
bimgs = np.zeros((sizey, sizex, number_files_to_loop ))
# images loaded in - unmodifed
img_save = np.zeros((sizey, sizex, 3, number_files_to_loop ))
# boolean (yes - edge)
edge = np.zeros((sizey, sizex, number_files_to_loop))
# each pixel's distance from sun
distance_from_sun = np.zeros((sizey, sizex, number_files_to_loop))
# number of pixels from edge of the cloud
length_inside_cloud = np.zeros((sizey, sizex, number_files_to_loop))
# total length of cloud object along the irradiance vector
length_of_cloud_ray = np.zeros((sizey, sizex, number_files_to_loop))
# relative length
fraction_inside_cloud = np.zeros((sizey, sizex, number_files_to_loop))
# time to perform preprocessing steps
how_long = np.zeros((number_files_to_loop, 1))

#%%

for l in range(number_files_to_loop):
    # D: indiv_frames is a directory with files in it. files_mask is list 
    # of a files from directory that have been masked. So indiv_frames + 
    # files_mask is a string that gives the name of one of the masked files.
    # Apparently this is an image label? Gives some creedence to previous 
    # question.  
    img_label = cv2.imread(indiv_frames + files_mask[l])
    img = cv2.imread(indiv_frames + files_rend[l])
    img = cv2.resize(img, (sizex, sizey))
    img_label = cv2.resize(img_label, (sizex, sizey))
    # D: img_save is sizey x sizex x 3 x number_files_to_loop. Is this
    # image dimension x rgb x number of training examples? Is each entry
    # in img and img_label a tuple or another array?
    img_save[:,:,:, l] = img
    
    # D: What is the purpose of the kmeans clustering?
    
    #draw edges.
    
    #edge[edge == 0] = np.nan
    
    # clustering the colors together.
    # setting the initial cluster points so that the clustering is 
    # consistent between all images
    # Converting labels from rgb to labels (1,2,3,4)
    init = np.array([[0, 0, 0], [7, 228, 95],
                    [105, 124, 109], [190, 205, 193]])
    kmeans = KMeans(
                    n_clusters = 4, random_state = 3, 
                    init = init, n_init = 1
                )
    # D: According to documentation cv2.split is time expensive. 
    r, g, b = cv2.split(img_label)
    X = np.array([r.flatten(), g.flatten(), b.flatten()]).T
    # reshaping cluster into 2d space
    Y = kmeans.fit_predict(X)
    label[:,:,l] = np.reshape(Y, (sizey, sizex))
    
#    plt.imshow(label[:,:,l])
#    print('3-side, 2-base, 1-sky, 0-nip')
#    plt.colorbar()
#    plt.show()
#    input('ya or na')
    
    
    labele = np.zeros((sizey, sizex, 4))
    #looking north-east,south,west of each point for edge detection
    labele[:-1,:, 0] = label[1:,:,l]
    labele[:, :-1, 1] = label[:, 1:, l]
    labele[1:,:, 2] = label[:-1, :, l]
    labele[:,1:, 3] = label[:, :-1, l]
    
    
    for i in range(sizey):
        for j in range(sizex):
            unique_vals = np.unique(labele[i, j, :])
            if np.size(unique_vals)>1:
                if np.size(np.unique(np.isin(unique_vals, [3,1]))) == 1: #side-sky
                    
                    edge[i,j,l] = 1
                elif np.size(np.unique(np.isin(unique_vals, [2,1]))) == 1: #base-sky
                    edge[i,j,l] = 1#2
                elif np.size(np.unique(np.isin(unique_vals, [2,3]))) == 1: #base-edge
                    edge[i,j,l] = 2#3
                else:
                    edge[i,j,l] = 0
    
    #convert the image into brightness space
    bimg = np.sqrt(np.sum((img/255)**2, axis = 2))
    bimg /= np.max(bimg)
    
    bimgs[:, :, l] = bimg
    #convert to sun centered polar coordinate system
    sun_coords_x = 110
    sun_coords_y = 110
    
    radial_coords = np.zeros(np.shape(bimg))
    azimuthal_coords = np.zeros(np.shape(bimg))
    
    f = sp.interpolate.interp2d(range(sizex), range(sizey), bimg)
    for i in range(np.shape(bimg)[1]):
        for j in range(np.shape(bimg)[0]):
            #calculate the radial and azimuithal coordinates on the grid of the image
            radial_coords[j, i] = np.sqrt((i - sun_coords_x)**2 + (j - sun_coords_y)**2)
            azimuthal_coords[j, i] = np.arctan2(j - sun_coords_y, i - sun_coords_x)
            
    
    
    #calculate the gradient at the pixel in question
    Dr = 10
    range_array = np.linspace(-10,10,21)
    range_array_0 = np.arange(0, 240,1)
    
    
    t1 = time.time()
    print(l/number_files_to_loop)

    for i in range(sizex):

#        print(i)

        for j in range(sizey):
   
            if bimg[j, i] == 0:

                continue

            elif (np.sqrt((i - sizex/2)**2+(j-sizey/2)**2)<(sizex/2-5)):
    #            print('ya')
                #interpolate along the radial direction             
                r_pixel = radial_coords[j, i]
                th_pixel = azimuthal_coords[j, i]
                range_array2 = range_array + r_pixel
                #finding the coordinates of the pixel
                X0 = r_pixel * np.cos(th_pixel) + sun_coords_x
                Y0 = r_pixel * np.sin(th_pixel) + sun_coords_y
                #find a range of pixels near the origin pixel
                X1 = range_array2 * np.cos(th_pixel) + sun_coords_x
                Y1 = range_array2 * np.sin(th_pixel) + sun_coords_y
                
                #calculate the pixels along the suncs ray
                X_along_sun_ray = range_array_0 * np.cos(th_pixel) + sun_coords_x
                Y_along_sun_ray = range_array_0 * np.sin(th_pixel) + sun_coords_y
                X_along_sun_rayr = np.round(X_along_sun_ray)
                Y_along_sun_rayr = np.round(Y_along_sun_ray)
                
                #calculate weighted average at the point
                X1r = np.round(X1)
                Y1r = np.round(Y1)
                values_along_radial = np.zeros(len(X1))
                
                #calculate the value of the pixel using interpolation
                t_11 = time.time()
#                for k in range(len(X1)):
                
#                    X_around_pixel, Y_around_pixel = np.meshgrid(np.arange(X1r[k]-2,X1r[k]+2), np.arange(Y1r[k]-2,Y1r[k]+2))
#                    D = np.sqrt((X_around_pixel - X1[k])**2 + (Y_around_pixel - Y1[k])**2)
#                    W = np.exp(-D)
                    
#                    Z = bimg[int(Y1r[k]-2):int(Y1r[k]+2), int(X1r[k]-2):int(X1r[k]+2)]
                try:
#                        f = sp.interpolate.interp2d(X_around_pixel, Y_around_pixel, Z)

                    
#                        values_along_radial[k] = np.sum(W*bimg[int(Y1r[k]-2):int(Y1r[k]+2), int(X1r[k]-2):int(X1r[k]+2)])/np.sum(W)
                    values_along_radial = f(X1, Y1)[0,:]
                except:
                        values_along_radial[:] = 0
                t_22 = time.time()
                how_long[l] = how_long[l]+t_22-t_11
                #calculate the value of the edge along the sun's ray  
                edges_along_radial = np.zeros(len(X_along_sun_ray))
                mask = (Y_along_sun_ray<sizey-5) & (Y_along_sun_rayr>5) & (X_along_sun_rayr>5) & (X_along_sun_rayr<sizex-5)
                X_along_sun_rayr = X_along_sun_rayr[mask]
                Y_along_sun_rayr = Y_along_sun_rayr[mask]
#                print(X_along_sun_rayr)
#                for k in range(len(X_along_sun_rayr)):
                edges_along_radial = edge[np.int0(Y_along_sun_rayr), np.int0(X_along_sun_rayr), l]

                        
                        
#                print(edges_along_radial)       
                inds = np.where(edges_along_radial == 1)[0]
                #calculate how far inside a cloud the pixel is
                try:
                    inds2 = inds - r_pixel
                    top  = np.min(inds2[inds2>=0])
                    bot = np.max(inds2[inds2 <0])
                    length_inside_cloud[j,i,l] = np.abs(bot)
    #                if length_inside_cloud[j,i,l] > 0:
    #                    print(length_inside_cloud[j,i,l] )
                    length_of_cloud_ray[j,i,l] = top-bot
                    fraction_inside_cloud[j,i,l] = np.abs(bot)/(top-bot)
                except:
                    length_inside_cloud[j,i,l] = 0
    #                if length_inside_cloud[j,i,l] > 0:
    #                    print(length_inside_cloud[j,i,l] )
                    length_of_cloud_ray[j,i,l] = 0
                    fraction_inside_cloud[j,i,l] = 0
                #assign all of the results into array for use later 
                distance_from_sun[j, i, l] = r_pixel
                whole_gradient[j, i, l] = np.mean(np.gradient(values_along_radial))
                front_gradient[j, i, l] = np.mean(np.gradient(values_along_radial)[11:])
                rear_gradient[j, i, l] = np.mean(np.gradient(values_along_radial)[:9])
                middle_gradient[j, i, l] = np.mean(np.gradient(values_along_radial)[5:15])
                average_near_pixel[j, i, l] = np.mean(bimg[int(Y0-4):int(Y0+4), int(X0-4):int(X0+4)])
                
           
    t2 = time.time()
    print(t2-t1)
             
#%%
np.save('/data/lasso/sims/20160611/bimgs', bimgs)
np.save('/data/lasso/sims/20160611/whole_gradient.npy', whole_gradient)
np.save('/data/lasso/sims/20160611/front_gradient.npy', front_gradient)
np.save('/data/lasso/sims/20160611/rear_gradient.npy', front_gradient)
np.save('/data/lasso/sims/20160611/middle_gradient.npy', middle_gradient)
np.save('/data/lasso/sims/20160611/average_near_pixel', average_near_pixel)
np.save('/data/lasso/sims/20160611/img_save.npy', img_save)
np.save('/data/lasso/sims/20160611/length_of_cloud_ray.npy', length_of_cloud_ray)
np.save('/data/lasso/sims/20160611/fraction_inside_cloud.npy', fraction_inside_cloud)
np.save('/data/lasso/sims/20160611/length_inside_cloud.npy', length_inside_cloud)
np.save('/data/lasso/sims/20160611/label.npy', label)

#%%
bimgs = np.load('/data/lasso/sims/20160611/bimgs.npy')
whole_gradient = np.load('/data/lasso/sims/20160611/whole_gradient.npy')
front_gradient = np.load('/data/lasso/sims/20160611/front_gradient.npy')
rear_gradient = np.load('/data/lasso/sims/20160611/rear_gradient.npy')
middle_gradient = np.load('/data/lasso/sims/20160611/middle_gradient.npy')
average_near_pixel = np.load('/data/lasso/sims/20160611/average_near_pixel.npy')
img_save = np.load('/data/lasso/sims/20160611/img_save.npy')
length_of_cloud_ray = np.load('/data/lasso/sims/20160611/length_of_cloud_ray.npy')
fraction_inside_cloud = np.load('/data/lasso/sims/20160611/fraction_inside_cloud.npy')
length_inside_cloud = np.load('/data/lasso/sims/20160611/length_inside_cloud.npy')
label = np.load('/data/lasso/sims/20160611/label.npy')

#%%
def objective_shuffle(la):
    num2 = np.sum(la == 2)
    num3 = np.sum(la == 3)
    la2 = np.zeros(np.shape(la))
    if num2>num3:
        inds2 = np.where(la == 2)[0]
        np.random.shuffle(inds2)
        inds2shn = inds2[0:num3]
        inds3 = np.where(la == 3)
        la2[inds2shn] = 2
        la2[inds3] = 3
    if num3>num2:
        inds3 = np.where(la == 3)[0]
        np.random.shuffle(inds3)
        inds3shn = inds3[0:num2]
        inds2 = np.where(la == 2)
        la2[inds3shn] = 3
        la2[inds2] = 2
    return la2

def cleanup(result, twoCorr, threeCorr):
    #looking around the pixel for modal averaging
    for i in np.random.permutation(np.arange(0,sizex)):
        for j in np.random.permutation(np.arange(0,sizey)):
            test = np.zeros((8))
            try:
                if result[j,i] == 2:
                    test[0] = result[j,i+1]
                    test[1] = result[j+1,i+1]
                    test[2] = result[j-1,i+1]
                    test[3] = result[j+1,i]
                    test[4] = result[j-1,i]
                    test[5] = result[j+1,i-1]
                    test[6] = result[j,i-1]
                    test[7] = result[j-1,i-1]
                if np.sum(test == 3) > twoCorr:
                    result[j,i] = 3
                    
                if result[j,i] == 3:
                    test[0] = result[j,i+1]
                    test[1] = result[j+1,i+1]
                    test[2] = result[j-1,i+1]
                    test[3] = result[j+1,i]
                    test[4] = result[j-1,i]
                    test[5] = result[j+1,i-1]
                    test[6] = result[j,i-1]
                    test[7] = result[j-1,i-1]
                if np.sum(test == 2) > threeCorr:
                    result[j,i] = 2
            except:
                Nick = 'Tired'
    return result
        
#%% Train the model

#unravel all of the arrays

number_to_train_with = 450

wg = np.ravel(whole_gradient[:,:,0:number_to_train_with])
fg = np.ravel(front_gradient[:,:,0:number_to_train_with])
rg = np.ravel(rear_gradient[:,:,0:number_to_train_with])
mg = np.ravel(middle_gradient[:,:,0:number_to_train_with])
anp = np.ravel(average_near_pixel[:,:,0:number_to_train_with])
bi = np.ravel(bimgs[:,:,0:number_to_train_with])**4
rr = np.ravel(img_save[:,:,0, :][:,:,0:number_to_train_with])
bb = np.ravel(img_save[:,:,1, :][:,:,0:number_to_train_with])
gg = np.ravel(img_save[:,:,2, :][:,:,0:number_to_train_with])
lcr = np.ravel(length_of_cloud_ray[:,:,0:number_to_train_with])
fic = np.ravel(fraction_inside_cloud[:,:,0:number_to_train_with])
lic = np.ravel(length_inside_cloud[:,:,0:number_to_train_with])

# create a new array with radius from the zenith


Xcoords = np.arange(0,sizex)
Ycoords = np.arange(0,sizey)

# limit calculation to only contain the values inside the rendered region
# D: Does the "rendered region" mean the non-empty pixels? This would be
# helpful and simplify the training
XXC, YYC = np.meshgrid(Xcoords, Ycoords)
# D: Creates array with each pixel assigned radius from zenith
RRC = np.sqrt((XXC-sizex/2)**2+(YYC-sizey/2)**2)
# D: This doesn't seem to be used anywhere
RRRC = np.ravel(RRC, order = 'C')
XXXC = np.ravel(XXC, order = 'C')
YYYC = np.ravel(YYC, order = 'C')
RRRRC = np.zeros((sizey,sizex, number_to_train_with))

for i in range(number_to_train_with):
    RRRRC[:,:,i] = RRC

RC = np.ravel(RRRRC, order = 'C')

# D: labels to train with
la = np.ravel(label[:,:,0:number_to_train_with])

#we dont want it to go over ambiguous areas

X = np.squeeze(np.array((wg, fg, rg, mg, anp, bi, rr, bb, gg, fic, RC))).T
#X = np.squeeze(np.array((mg, bi, lcr, fic, lic, RC))).T
#Xt = np.squeeze(np.array((mgt, bit, lcrt, fict, lict, RCt))).T

labels = ['wg', 'fg', 'rg', 'mg', 'anp', 'bi', 'rr', 'bb', 'gg', 'lcr', 'fic', 'lic', 'RC']

#setup the coordinates 

# D: Isolate base and side pixels in labels
mask0 = (la > 1)

# D: circle/annulus radii
r1 = np.sqrt(1/(1+np.sqrt(2)))*sizex/2
r2 = np.sqrt(2/(1+np.sqrt(2)))*sizex/2



# grab only the important values
# D: innermost circle
mask1 = (RC >= 0) & (RC < r1)
# D: first annulus
mask2 = (RC >= r1) & (RC < r2) 
# D: second annulus
mask4 = (RC >= r2)&(RC < sizex/2)

# D: Grab training data
# D: base and sides inside inner most circle
X1 = X[mask0 & mask1, :]
# D: base and sides inside first annulus
X2 = X[mask0 & mask2, :]
# D: base and sides inside second annulus
X4 = X[mask0 & mask4, :]

# D: Grab training labels. Corresponds to above
la1 = la[mask1 & mask0]
la2 = la[mask2 & mask0]
la4 = la[mask4 & mask0]


## creating equal numbers of labels
#mask11 = objective_shuffle(la1) > 1
mask22 = objective_shuffle(la2) > 1
#mask44 = objective_shuffle(la4) > 1
##
#X1 = X1[mask11]
X2 = X2[mask22]
#X4 = X4[mask44]
##
#la1 = la1[mask11]
la2 = la2[mask22]
#la4 = la4[mask44]
##


#scale the data in preparation for models
# D: Do we need so many scaler objects? One should work for all.
# In fact, we might not need any. Can scale data by max rgb value 255
scaler1 = StandardScaler()
scaler2 = StandardScaler()
#scaler3 = StandardScaler()
scaler4 = StandardScaler()

scaler1.fit(X1)
scaler2.fit(X2)
scaler4.fit(X4)

X1S = scaler1.transform(X1)
X2S = scaler2.transform(X2)
X4S = scaler4.transform(X4)

#assign the initial conditions, hyperparameters, and model.

#tree = ExtraTreesClassifier(n_estimators=4, max_depth=None, min_samples_split=4, 
#                            random_state=0, n_jobs = 6, bootstrap=False,
#                            class_weight=None, criterion='gini', max_features='auto',
#                            max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
#                            min_samples_leaf=1, min_weight_fraction_leaf=0, oob_score=False, 
#                            verbose=0, warm_start=True)
#tree = MLPClassifier(hidden_layer_sizes=(100, ), activation='logistic', solver='adam', 
#                     alpha=0.001, batch_size='auto', learning_rate='constant', 
#                     learning_rate_init=0.001, power_t=0.5, max_iter=120, shuffle=True, 
#                     random_state=None, tol=0.0001, verbose=False, warm_start=True, 
#                     momentum=0.9, nesterovs_momentum=True, early_stopping=True, 
#                     validation_fraction=0.1, beta_1=0.9, beta_2=0.995, epsilon=1e-08, 
#                     n_iter_no_change=10) #this one seems promising


#tree1 = KNeighborsClassifier(n_neighbors=20, weights = 'distance', leaf_size=100, n_jobs = 6) #works fairly well
#tree2 = KNeighborsClassifier(n_neighbors=20, weights = 'distance', leaf_size=100, n_jobs = 6) #works fairly well
#tree3 = KNeighborsClassifier(n_neighbors=20, weights = 'distance', leaf_size=100, n_jobs = 6) #works fairly well
#tree4 = KNeighborsClassifier(n_neighbors=20, weights = 'distance', leaf_size=100, n_jobs = 6) #works fairly well


n_est = 50
#tree1 = ExtraTreesClassifier(n_estimators=n_est, max_depth=None, min_samples_split=4, 
#                            random_state=None, n_jobs = 8, bootstrap=False,
#                            class_weight=None, criterion='gini', max_features='auto',
#                            max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
#                            min_samples_leaf=3, min_weight_fraction_leaf=0, oob_score=False, 
#                            verbose=0, warm_start=True)

tree2 = ExtraTreesClassifier(n_estimators=n_est, max_depth=None, min_samples_split=4, 
                            random_state=None, n_jobs = 8, bootstrap=False,
                            class_weight=None, criterion='gini', max_features='auto',
                            max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                            min_samples_leaf=3, min_weight_fraction_leaf=0, oob_score=False, 
                            verbose=0, warm_start=True)
#
#tree3 = ExtraTreesClassifier(n_estimators=n_est, max_depth=None, min_samples_split=4, 
#                            random_state=0, n_jobs = 6, bootstrap=False,
#                            class_weight=None, criterion='gini', max_features='auto',
#                            max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
#                            min_samples_leaf=1, min_weight_fraction_leaf=0, oob_score=False, 
#                            verbose=0, warm_start=True)
#
tree4 = ExtraTreesClassifier(n_estimators=n_est, max_depth=None, min_samples_split=4, 
                            random_state=None, n_jobs = 8, bootstrap=False,
                            class_weight=None, criterion='gini', max_features='auto',
                            max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                            min_samples_leaf=3, min_weight_fraction_leaf=0, oob_score=False, 
                            verbose=0, warm_start=True)

#tree1 = DecisionTreeClassifier()
#tree2 = DecisionTreeClassifier()
#tree4 = DecisionTreeClassifier()

smo = 0.3
tree1 = GaussianNB(var_smoothing = smo)
#tree2 = GaussianNB(var_smoothing = smo)
#tree4 = GaussianNB(var_smoothing = smo)

#tree1 = LinearSVC()
#tree2 = LinearSVC()
#tree3 = LinearSVC()
#tree4 = LinearSVC()


#tree1 = SGDClassifier(loss='hinge', penalty='l2', alpha=0.001, 
#                      l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, 
#                      shuffle=True, verbose=0, epsilon=0.1, n_jobs=8, random_state=None, 
#                      learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, 
#                      validation_fraction=0.1, n_iter_no_change=5, class_weight=None, 
#                      warm_start=True, average=False)
#tree2 = SGDClassifier(loss='hinge', penalty='l2', alpha=0.001, 
#                      l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, 
#                      shuffle=True, verbose=0, epsilon=0.1, n_jobs=8, random_state=None, 
#                      learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, 
#                      validation_fraction=0.1, n_iter_no_change=5, class_weight=None, 
#                      warm_start=True, average=False)
#tree4 = SGDClassifier(loss='hinge', penalty='l2', alpha=0.001, 
#                      l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, 
#                      shuffle=True, verbose=0, epsilon=0.1, n_jobs=8, random_state=None, 
#                      learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, 
#                      validation_fraction=0.1, n_iter_no_change=5, class_weight=None, 
#                      warm_start=True, average=False)



# Train da model.

tree1.fit(X1S, la1)
tree2.fit(X2S, la2)
##tree3.fit(X3S, la3)
tree4.fit(X4S, la4)

#%%
#fit the data with the label


#testing on one of the images trained on

num = 460 #number of the image
#test
for num in [460]:#range(number_to_train_with):
    result = np.zeros((sizey,sizex))
    result1 = np.zeros((sizey,sizex))
    result2 = np.zeros((sizey,sizex))
    result3 = np.zeros((sizey,sizex))
    
    #train the model in the four areas
    
    
    #identify the training 'picture'
    wgt = np.ravel(whole_gradient[:,:,num])
    fgt = np.ravel(front_gradient[:,:,num])
    rgt = np.ravel(rear_gradient[:,:,num])
    mgt = np.ravel(middle_gradient[:,:,num])
    anpt = np.ravel(average_near_pixel[:,:,num])
    bit = np.ravel(bimgs[:,:,num])**4
    rrt = np.ravel(img_save[:,:,0, :][:,:,num])
    bbt = np.ravel(img_save[:,:,1, :][:,:,num])
    ggt = np.ravel(img_save[:,:,2, :][:,:,num])
    lcrt = np.ravel(length_of_cloud_ray[:,:,num])
    fict = np.ravel(fraction_inside_cloud[:,:,num])
    lict = np.ravel(length_inside_cloud[:,:,num])
    RCt = np.ravel(RRC, order = 'C')
    lat = np.ravel(label[:,:,num])
    
    Xt = np.squeeze(np.array((wgt, fgt, rgt, mgt, anpt, bit, rrt, bbt, ggt, fict, RCt))).T
    
    
    #eventually I'll rewrite this to use arrays instead of repeating things four times
    
    mask0t = (lat > 1)
    mask1t = (RCt >= 0) & (RCt < r1)
    mask2t = (RCt >= r1) & (RCt < r2) 
    #mask3t = (RCt >= sizex/(2*2)) & (RCt < sizex/(1.41*2)) 
    mask4t = (RCt >= r2) & (RCt < sizex/2)
    
    
    
    X1t = Xt[mask0t & mask1t, :]
    X2t = Xt[mask0t & mask2t, :]
    #X3t = Xt[mask0t & mask3t, :]
    X4t = Xt[mask0t & mask4t, :]
    
    X1tS = scaler1.transform(X1t)
    X2tS = scaler2.transform(X2t)
    #X3tS = scaler3.transform(X3t)
    X4tS = scaler4.transform(X4t)
    
    
    
    
    
    
    
    # D: What are these for?
    # Seems to be meshgrid with only certain annuli
    XXXC1 = XXXC[mask0t & mask1t]
    YYYC1 = YYYC[mask0t & mask1t]
    XXXC2 = XXXC[mask0t & mask2t]
    YYYC2 = YYYC[mask0t & mask2t]
    #XXXC3 = XXXC[mask0t & mask3t]
    #YYYC3 = YYYC[mask0t & mask3t]
    XXXC4 = XXXC[mask0t & mask4t]
    YYYC4 = YYYC[mask0t & mask4t]
    
    lat1 = lat[mask0t & mask1t]
    lat2 = lat[mask0t & mask2t]
    lat4 = lat[mask0t & mask4t]
    
    
    prediction1 = tree1.predict_proba(X1tS)[:,1]
    for i in range(len(XXXC1)):
        result1[YYYC1[i], XXXC1[i]] = prediction1[i]
        
    result1 = cleanup(result1, 4, 6)
       
    prediction2 = tree2.predict_proba(X2tS)[:,1]
    for i in range(len(XXXC2)):
        result2[YYYC2[i], XXXC2[i]] = prediction2[i]
    result2 = cleanup(result2, 6, 4)
    #prediction3 = tree3.predict(X3tS)
    #for i in range(len(XXXC3)):
    #    result[YYYC3[i],XXXC3[i]] = prediction3[i]
    
    prediction4 = tree4.predict_proba(X4tS)[:,1]
    for i in range(len(XXXC4)):
        result3[YYYC4[i],XXXC4[i]] = prediction4[i]
    result3 = cleanup(result3, 4, 6)
    
    
    #result = np.ravel(result)
    #result1 = np.ravel(result1)
    #result2 = np.ravel(result2)
    #result3 = np.ravel(result3)
    
    result[result1 !=0] = result1[result1 !=0]
    result[result2 !=0] = result2[result2 !=0]
    result[result3 !=0] = result3[result3 !=0]
    
    #np.reshape(result, (sizey, sizex))
    
    
    
    
    #cleaning up the results
    
    
                    
    result[result == 0] = np.nan


#%%

# D: Seems like most of this cell is a duplicating cell above.
# Can perhaps delete? 

#testing on one of the images trained on

num = 460 #number of the image
#test

result = np.zeros((sizey,sizex))
result1 = np.zeros((sizey,sizex))
result2 = np.zeros((sizey,sizex))
result3 = np.zeros((sizey,sizex))

#train the model in the four areas


#identify the training 'picture'
wgt = np.ravel(whole_gradient[:,:,num])
fgt = np.ravel(front_gradient[:,:,num])
rgt = np.ravel(rear_gradient[:,:,num])
mgt = np.ravel(middle_gradient[:,:,num])
anpt = np.ravel(average_near_pixel[:,:,num])
bit = np.ravel(bimgs[:,:,num])**4
rrt = np.ravel(img_save[:,:,0, :][:,:,num])
bbt = np.ravel(img_save[:,:,1, :][:,:,num])
ggt = np.ravel(img_save[:,:,2, :][:,:,num])
lcrt = np.ravel(length_of_cloud_ray[:,:,num])
fict = np.ravel(fraction_inside_cloud[:,:,num])
lict = np.ravel(length_inside_cloud[:,:,num])
RCt = np.ravel(RRC, order = 'C')
lat = np.ravel(label[:,:,num])

Xt = np.squeeze(np.array((wgt, fgt, rgt, mgt, anpt, bit, rrt, bbt, ggt, fict, RCt))).T


#eventually I'll rewrite this to use arrays instead of repeating things four times

mask0t = (lat > 1)
mask1t = (RCt >= 0) & (RCt < r1)
mask2t = (RCt >= r1) & (RCt < r2) 
#mask3t = (RCt >= sizex/(2*2)) & (RCt < sizex/(1.41*2)) 
mask4t = (RCt >= r2) & (RCt < sizex/2)

# D: numpy does not support non-rectangular arrays. Can use dictionary?
# Like this

# Mask = {}
# Mask['bs'] = (lat > 1)
# Mask['c1'] = (RCt >= 0) & (RCt < r1)
# Mask['a1'] = (RCt >= r1) & (RCt < r2)
# Mask['a2'] = (RCt >= r2) & (RCt < sizex/2)



X1t = Xt[mask0t & mask1t, :]
X2t = Xt[mask0t & mask2t, :]
#X3t = Xt[mask0t & mask3t, :]
X4t = Xt[mask0t & mask4t, :]

X1tS = scaler1.transform(X1t)
X2tS = scaler2.transform(X2t)
#X3tS = scaler3.transform(X3t)
X4tS = scaler4.transform(X4t)








XXXC1 = XXXC[mask0t & mask1t]
YYYC1 = YYYC[mask0t & mask1t]
XXXC2 = XXXC[mask0t & mask2t]
YYYC2 = YYYC[mask0t & mask2t]
#XXXC3 = XXXC[mask0t & mask3t]
#YYYC3 = YYYC[mask0t & mask3t]
XXXC4 = XXXC[mask0t & mask4t]
YYYC4 = YYYC[mask0t & mask4t]

lat1 = lat[mask0t & mask1t]
lat2 = lat[mask0t & mask2t]
lat4 = lat[mask0t & mask4t]

#rewrap into the XxY array shape
prediction1 = tree1.predict(X1tS)
for i in range(len(XXXC1)):
    result1[YYYC1[i], XXXC1[i]] = prediction1[i]
    
result1 = cleanup(result1, 4, 6)
   
prediction2 = tree2.predict(X2tS)
for i in range(len(XXXC2)):
    result2[YYYC2[i], XXXC2[i]] = prediction2[i]
result2 = cleanup(result2, 6, 4)
#prediction3 = tree3.predict(X3tS)
#for i in range(len(XXXC3)):
#    result[YYYC3[i],XXXC3[i]] = prediction3[i]

prediction4 = tree4.predict(X4tS)
for i in range(len(XXXC4)):
    result3[YYYC4[i],XXXC4[i]] = prediction4[i]
result3 = cleanup(result3, 4, 6)


#result = np.ravel(result)
#result1 = np.ravel(result1)
#result2 = np.ravel(result2)
#result3 = np.ravel(result3)

result[result1 !=0] = result1[result1 !=0]
result[result2 !=0] = result2[result2 !=0]
result[result3 !=0] = result3[result3 !=0]

#np.reshape(result, (sizey, sizex))




#cleaning up the results


                
result[result == 0] = np.nan





prediction = np.ravel(result)
print('% correct: ', np.sum(prediction[mask0t] == lat[mask0t])/np.size(lat[mask0t]))
print('% model 1: ', np.sum(lat1[lat1>1] == prediction1)/np.size(lat1[lat1>1]) )
print('% model 2: ', np.sum(lat2[lat2>1] == prediction2)/np.size(lat2[lat2>1]) )
print('% model 4: ', np.sum(lat4[lat4>1] == prediction4)/np.size(lat4[lat4>1]) )

#showing (hopefully) pretty pictures


 #%%
ang = np.arange(0,2*np.pi, 2*np.pi/360)
Circlex = np.cos(ang)
Circley = np.sin(ang)

plt.figure(figsize = (15,15))
plt.imshow(result, vmin=0, vmax=3)

#mask1t = (RCt >= 0)&(RCt<sizex/(r1) )
#mask2t = (RCt >= sizex/(r1))&(RCt<sizex/(2*2)) 
#mask3t = (RCt >= sizex/(2*2))&(RCt<sizex/(1.41*2)) 
#mask4t = (RCt >= sizex/(1.41*2))&(RCt<sizex/2)

plt.title('Naive Bayes Classification')
plt.plot(sizex/2+ r1*Circlex, sizey/2+ r1*Circley, c = 'k')
#plt.plot(sizex/2+ sizex/(2*2)*Circlex, sizey/2+ sizex/(2*2)*Circley, c = 'k')
plt.plot(sizex/2+ r2*Circlex, sizey/2+ r2*Circley, c = 'k')
plt.plot(sizex/2+ sizex/2*Circlex, sizey/2+ sizex/2*Circley, c = 'k')
plt.colorbar()
#plt.savefig('/data/lasso/sims/20160611/figures/NBC.png')
 

plt.figure(figsize = (15,15))
plt.title('Ground Truth')
plt.imshow(label[:,:,num], vmin=0, vmax=3)
plt.colorbar()
#plt.savefig('/data/lasso/sims/20160611/figures/GT.png')

#edge[edge == 0] = np.nan
#edge[edge == 1] = 4
#edge[edge == 2] = 5
#plt.imshow(edge[:,:,0])
#plt.colorbar()
#unwrap_prediction = np.reshape(prediction, (sizey, sizex, number_files_to_loop))
#
#plt.figure(figsize = (15,15))
#plt.imshow(unwrap_prediction[:,:,0])
#
#plt.imshow(edge[:,:,0])
#plt.figure(figsize = (15,15))
#plt.imshow(label[:,:,0])



