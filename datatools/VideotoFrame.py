#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 16:18:33 2018

@author: nick barron
"""

import os
import cv2

'''
User options: 
    MASKED  = True 
        returns masked images in directory
    MASKED = False 
        returns 'pretty' images
    
    DIRECTORY = 'string' 
        please modify this string to the parent directory of the videos
'''

MASKED = True
DIRECTORY = '/data/lasso/sims/20160625/videos'
LANDINGNAME = '/data/lasso/sims/20160625/indiv_frames'

xcoords = [6400, 9600, 12800, 16000, 19200]

if MASKED == False:
    name_0 = 'TSI_R '
else:
    name_0 = 'TSI_R_MASKED '

timestamps = os.listdir(DIRECTORY)
for timestamp in timestamps:
    for xcoord in xcoords:
        newdir = DIRECTORY + '/' + timestamp + '/'
        name_1 = '(along x = ' + str(xcoord) + ').avi'
        
        video = cv2.VideoCapture(newdir + name_0 + name_1)
        success = True
        
        for i in range(111):
#            print('Frame number = ', i)
            ycoord = i * 150 #m 
            
            success, img = video.read()
            if type(img) == None:
                raise NameError('Files are not in DIRECTORY, rename DIRECTORY to reflect correct directory')
            
            '''
            Several options may exist at this point, simply saving the data or 
            utilizing some sort of unwrapping algorithm. Or, at this point, the 
            image may be fed to the machine learning algorithm
            '''
            
            image_landing = (
                LANDINGNAME + '/' + timestamp + '_' + str(xcoord) 
                + "_" + name_0[:-1] + '_' + str(i) +'.png'
                )
            cv2.imwrite(image_landing, img)
        