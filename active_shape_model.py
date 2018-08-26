# -*- coding: utf-8 -*-
"""
Created on Wen Aug 23 18:06:19 2018

@author: manoj bhor
"""

import numpy as np
import pandas as pd  
from scipy.linalg import norm
from math import atan
from math import sin, cos



# FEW FUNCTIONS FOR ALLIGNMENT
def get_translation(shape):
  '''
  Calculates a translation for x and y
  axis that centers shape around the
  origin
  Args:
    shape(2n x 1 NumPy array) an array 
    containing x coodrinates of shape
    points as first column and y coords
    as second column
   Returns:
    translation([x,y]) a NumPy array with
    x and y translationcoordinates
  '''
  mean_x = np.mean(shape[::2]).astype(np.int)
  mean_y = np.mean(shape[1::2]).astype(np.int)
  
  return np.array([mean_x, mean_y])

def translate(shape):
  '''
  Translates shape to the origin
  Args:
    shape(2n x 1 NumPy array) an array 
    containing x coodrinates of shape
    points as first column and y coords
    as second column
  '''
  mean_x, mean_y = get_translation(shape)
  shape[::2] -= mean_x
  shape[1::2] -= mean_y    
  
def get_rotation_scale(reference_shape, shape):
    '''
    Calculates rotation and scale
    that would optimally align shape
    with reference shape
    Args:
        reference_shape(2nx1 NumPy array), a shape that
        serves as reference for scaling and 
        alignment
        
        shape(2nx1 NumPy array), a shape that is scaled
        and aligned
        
    Returns:
        scale(float), a scaling factor
        theta(float), a rotation angle in radians
    '''
    
    a = np.dot(shape, reference_shape) / norm(reference_shape)**2
    
    #separate x and y for the sake of convenience
    ref_x = reference_shape[::2]
    ref_y = reference_shape[1::2]
    
    x = shape[::2]
    y = shape[1::2]
    
    b = np.sum(x*ref_y - ref_x*y) / norm(reference_shape)**2
    
    scale = np.sqrt(a**2+b**2)
    theta = atan(b / max(a, 10**-10)) #avoid dividing by 0
    
    return round(scale,1), round(theta,2)  

def get_rotation_matrix(theta):
    
    return np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

def scale(shape, scale):
    
    return shape / scale

def rotate(shape, theta):
    '''
    Rotates a shape by angle theta
    Assumes a shape is centered around 
    origin
    Args:
        shape(2nx1 NumPy array) an shape to be rotated
        theta(float) angle in radians
    Returns:
        rotated_shape(2nx1 NumPy array) a rotated shape
    '''
    
    matr = get_rotation_matrix(theta)
    
    #reshape so that dot product is eascily computed
    temp_shape = shape.reshape((-1,2)).T
    
    #rotate
    rotated_shape = np.dot(matr, temp_shape)
    
    return rotated_shape.T.reshape(-1)

def procrustes_analysis(reference_shape, shape):
    '''
    Scales, and rotates a shape optimally to
    be aligned with a reference shape
    Args:
        reference_shape(2nx1 NumPy array), a shape that
        serves as reference alignment
        
        shape(2nx1 NumPy array), a shape that is aligned
        
    Returns:
        aligned_shape(2nx1 NumPy array), an aligned shape
        translated to the location of reference shape
    '''
    #copy both shapes in caseoriginals are needed later
    temp_ref = np.copy(reference_shape)
    temp_sh = np.copy(shape)
 
    translate(temp_ref)
    translate(temp_sh)
    
    #get scale and rotation
    scale, theta = get_rotation_scale(temp_ref, temp_sh)
    
    #scale, rotate both shapes
    temp_sh = temp_sh / scale
    aligned_shape = rotate(temp_sh, theta)
    
    return aligned_shape

def procrustes_distance(reference_shape, shape):
    
    ref_x = reference_shape[::2]
    ref_y = reference_shape[1::2]
    
    x = shape[::2]
    y = shape[1::2]
    
    dist = np.sum(np.sqrt((ref_x - x)**2 + (ref_y - y)**2))
    
    return dist


def generalized_procrustes_analysis(shapes):
    '''
    Performs superimposition on a set of 
    shapes, calculates a mean shape
    Args:
        shapes(a list of 2nx1 Numpy arrays), shapes to
        be aligned
    Returns:
        mean(2nx1 NumPy array), a new mean shape
        aligned_shapes(a list of 2nx1 Numpy arrays), super-
        imposed shapes
    '''
    #initialize Procrustes distance
    current_distance = 0
    
    #initialize a mean shape
    mean_shape = shapes.iloc[0,:].values
    #mean_shape = shapes[0]
    
    num_shapes = len(shapes)
    
    #create array for new shapes, add 
    new_shapes = np.zeros(np.array(shapes).shape)
    
    
    while True:
        
        #add the mean shape as first element of array
        new_shapes[0] = mean_shape
        
        #superimpose all shapes to current mean
        for sh in range(1, num_shapes):
            new_sh = procrustes_analysis(mean_shape, shapes.iloc[sh].values)
            #new_sh = procrustes_analysis(mean_shape, shapes[sh])
            new_shapes[sh] = new_sh
        
        #calculate new mean
        new_mean = np.mean(new_shapes, axis = 0)
        
        new_distance = procrustes_distance(new_mean, mean_shape)
        
        #if the distance did not change, break the cycle
        if new_distance == current_distance:
            break
        
        #align the new_mean to old mean
        new_mean = procrustes_analysis(mean_shape, new_mean)
        
        #update mean and distance
        mean_shape = new_mean
        current_distance = new_distance
        
        
    return mean_shape, new_shapes 


#% load the dataset
face_images_db = np.load('input/face_images.npz')['face_images']
facial_keypoints_df = pd.read_csv('input/facial_keypoints.csv')


numMissingKeypoints = facial_keypoints_df.isnull().sum(axis=1)
allKeypointsPresentInds = np.nonzero(numMissingKeypoints == 0)[0]

faceImagesDB = face_images_db[:,:,allKeypointsPresentInds]
facialKeypointsDF = facial_keypoints_df.iloc[allKeypointsPresentInds,:].reset_index(drop=True)

(imHeight, imWidth, numImages) = faceImagesDB.shape
numKeypoints = facialKeypointsDF.shape[1] / 2

print('number of remaining images = %d' %(numImages))
print('image dimentions = (%d,%d)' %(imHeight,imWidth))
print('number of facial keypoints = %d' %(numKeypoints))


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
Xx_train, Xx_test, yy_train, yy_test = facialKeypointsDF.iloc[0:2000,:],facialKeypointsDF.iloc[2000:2141,:], faceImagesDB[:,:,0:2000], faceImagesDB[:,:,2000:2141]


# Aligning the points (Step 1 for Active Shape model)
mean_shape, X_train = generalized_procrustes_analysis(Xx_train)
    
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

# Applying PCA  (Step 2 for Active Shape model)
from sklearn.decomposition import PCA
pca = PCA(.98)
X_train = pca.fit_transform(X_train)
explained_variance = pca.explained_variance_ratio_

# Matching model points to target points
