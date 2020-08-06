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
from scipy import stats
from scipy.ndimage.measurements import label as bwlabel

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DIRECTORY = '/data/lasso/sims/20160611'
INDIV_FRAMES = DIRECTORY + '/indiv_frames/'
files = os.listdir(INDIV_FRAMES)
# split up the files into masked and rendered

mask = np.zeros(len(files))
for i in range(len(files)):
    if 'MASKED' in files[i]:
        mask[i] = 2
    else:
        mask[i] = 1
   
files = np.array(files)     
files_mask = files[mask == 2]
files_mask = np.sort(files_mask)
files_rend = files[mask == 1]
files_rend = np.sort(files_rend)

number_files_to_loop = 500

arr = np.arange(0, files_mask.size, 1)
np.random.shuffle(arr)
get_random_500 = arr[0:number_files_to_loop]

files_mask = files_mask[get_random_500]
files_rend = files_rend[get_random_500]

with open(DIRECTORY + 'images_used.txt', 'w') as filehandle:
    for filename in files_rend:
        filehandle.write('%s\n' % filename)

#img = cv2.imread(indiv_frames+files[1])

#load an image in
resize_ratio = 1
sizex = round(480*resize_ratio)
sizey = round(640*resize_ratio)

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

for l in range(number_files_to_loop):

    # read in image "label" and image itself and save image in in a numpy array
    img_label = cv2.imread(INDIV_FRAMES + files_mask[l])
    img = cv2.imread(INDIV_FRAMES + files_rend[l])
    img_save[:,:,:,l] = img
    
    # clustering the colors together.
    # setting the initial cluster points so that the clustering is 
    # consistent between all images
    # Converting labels from rgb to labels (1,2,3,4)
    init = np.array([[0, 0, 0], [7, 228, 95],
                    [105, 124, 109], [190, 205, 193]])/255
    kmeans = KMeans(n_clusters = 4, random_state = 3, 
                    init = init, n_init = 1)
    r, g, b = cv2.split(img_label)
    X = np.array([r.flatten(), g.flatten(), b.flatten()], dtype=np.float).T
    X /= 255
    Y = kmeans.fit_predict(X)
    label[:,:,l] = np.reshape(Y, (sizey, sizex))
    
    
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
    
    # calculate image "brightness" and scale it
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
    print(str(l / number_files_to_loop * 100) + 'percent complete')

    for i in range(sizex):
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

np.save(DIRECTORY + '/bimgs', bimgs)
np.save(DIRECTORY + '/whole_gradient.npy', whole_gradient)
np.save(DIRECTORY + '/front_gradient.npy', front_gradient)
np.save(DIRECTORY + '/rear_gradient.npy', front_gradient)
np.save(DIRECTORY + '/middle_gradient.npy', middle_gradient)
np.save(DIRECTORY + '/average_near_pixel', average_near_pixel)
np.save(DIRECTORY + '/img_save.npy', img_save)
np.save(DIRECTORY + '/length_of_cloud_ray.npy', length_of_cloud_ray)
np.save(DIRECTORY + '/fraction_inside_cloud.npy', fraction_inside_cloud)
np.save(DIRECTORY + '/length_inside_cloud.npy', length_inside_cloud)
np.save(DIRECTORY + '/label.npy', label)
