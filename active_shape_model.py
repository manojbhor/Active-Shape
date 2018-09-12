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

import matplotlib.pyplot as plt
from scipy.spatial import distance



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

#################
    
"""
def getEquidistantPoints(p1, p2, parts):
    return zip(np.linspace(p1[0], p2[0], parts+1), np.linspace(p1[1], p2[1], parts+1))
"""
def get_line(start, end):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end
 
    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    """
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
 
    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)
 
    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
 
    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
 
    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1
 
    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
 
    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
 
    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points

def normal_profile_pixels(data, image, ids, pix, llx1, lly1, llx2, lly2,llx3, lly3):
    '''
    '''
    #initialize Profile pixel distance
    l_x1, l_y1 = data.iloc[ids,llx1], data.iloc[ids,lly1]
    l_x2, l_y2 = data.iloc[ids,llx2], data.iloc[ids,lly2]
    l_x3, l_y3 = data.iloc[ids,llx3],data.iloc[ids,lly3]

    #print(l_x1, l_y1)
    #print(l_x2, l_y2)
    #print(l_x3, l_y3)

    a = np.sqrt(np.power((l_x1-l_x2), 2) + np.power((l_y1-l_y2), 2))
    b = np.sqrt(np.power((l_x1-l_x3), 2) + np.power((l_y1-l_y3), 2))
    c = np.sqrt(np.power((l_x2-l_x3), 2) + np.power((l_y2-l_y3), 2))
    
    x_feature_coords = ((l_x1*c) + (l_x2*b) + (l_x3*a))/(a+b+c)
    y_feature_coords = ((l_y1*c) + (l_y2*b) + (l_y3*a))/(a+b+c)

    print(x_feature_coords, y_feature_coords)
    

    """
    # show a random subset of images from the dataset
    box1 = image[:,:,ids]
    plt.close('all')
    fig, ax = plt.subplots()
   """

    
    x_coords = (x_feature_coords - l_x1) if abs((x_feature_coords - l_x1)) > 0.001 else (x_feature_coords - l_x1) *1000
    y_coords = (x_feature_coords - l_x1) if abs((x_feature_coords - l_x1)) > 0.001 else (x_feature_coords - l_x1) *1000 
    
    x_up_coords, y_up_coords = x_feature_coords + ((x_coords - l_x1)* 100), y_feature_coords + ((y_coords - l_y1)* 100)
    x_down_coords, y_down_coords = x_feature_coords - ((x_coords - l_x1)* 100), y_feature_coords - ((y_coords - l_y1)* 100)
    
    
    """   
    x1, y1 = [x_up_coords ,x_down_coords], [y_up_coords,y_down_coords ]
    #x1, y1 = [x_feature_coords - abs(x_feature_coords - l_x1),l_x1], [y_feature_coords - abs(y_feature_coords - l_y1), l_y1]
    print(x_feature_coords - l_x1, y_feature_coords - l_y1)
    ax.imshow(box1, cmap='gray');
    ax.plot(x1, y1)
    ax.scatter(l_x1,l_y1,c='r',s=12)
    ax.scatter(l_x2,l_y2,c='r',s=12)
    ax.scatter(l_x3,l_y3,c='r',s=12)
    ax.scatter(ox,oy,c='b',s=12)
    """
    
    """
    cords = list(getEquidistantPoints((x_feature_coords - abs(x_feature_coords - l_x1), y_feature_coords - abs(y_feature_coords - l_y1)), (ox,oy), 10))
    dt=np.dtype('float,float')
    cords = np.array(cords,dtype=dt)
    """
    cords = get_line((int(round(l_x1)),int(round(l_y1))), (int(round(x_up_coords)),int(round(y_up_coords))))
    cords1 = get_line((int(round(l_x1)),int(round(l_y1))), (int(round(x_down_coords)),int(round(y_down_coords))))
    
    #int(round(h))
    image1 = image[:,:,ids]
    #print(image1[cords[0]])
    #print(image1[cords1[6]])
    
    if pix == 5:
        mark_pixel= np.zeros([12], dtype=int)
    elif pix == 15:
        mark_pixel= np.zeros([32], dtype=int)
        co_points= np.zeros([32], dtype=float)
    co_points= []    
    # one side
    for i in range(pix):
        x,y = cords1[pix - i]
        co_points.append(cords1[pix - i])
        if x >= 96 or y >= 96 or x <= 0 or y<=0:
            mark_pixel[i] = 0  
        else:
            print(cords1[pix - i])
            mark_pixel[i] = image1[cords1[pix - i]]
        #print(cords1[5 - i])
    # another side    
    for i in range(pix + 2):
         x,y = cords[i]
         co_points.append(cords[i])
         if x >= 96 or y >= 96 or x <= 0 or y<=0:
            mark_pixel[pix + i] = 0  
         else:
            print(cords[i]) 
            mark_pixel[pix + i] = image1[cords[i]]
    #print(co_points) 
    
    xyz = np.diff(mark_pixel)    
    marked_der = xyz/np.sum(abs(xyz))    
    print(ids)
    print( mark_pixel)
    if pix == 15:
        
        return co_points, marked_der 
        #print(cords[i])
    
    return marked_der


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
# from sklearn.model_selection import train_test_split
Xx_train, Xx_test, yy_train, yy_test = facialKeypointsDF.iloc[0:2000,:],facialKeypointsDF.iloc[2000:2141,:], faceImagesDB[:,:,0:2000], faceImagesDB[:,:,2000:2141]

training_profile = np.zeros([Xx_train.shape[0], 15, 11])
test_profile = np.zeros([15, 31])
         #1
llx11, lly11 = 0,1
llx21, lly21 = 4,5
llx31,lly31 = 14, 15
#2
llx12, lly12 = 2, 3
llx22, lly22 = 8, 9
llx32, lly32 = 18, 19
#3
llx13, lly13 = 4, 5
llx23, lly23 = 0, 1
llx33, lly33 = 12, 13
#4
llx14, lly14 = 6, 7
llx24, lly24 = 0, 1
llx34, lly34 = 14, 15
#5
llx15, lly15 = 8,9
llx25, lly25 = 16,17
llx35, lly35 = 2,3
#6
llx16, lly16 = 10,11
llx26, lly26 = 2,3
llx36, lly36 = 18,19
#7
llx17, lly17 = 12,13
llx27, lly27 = 4,5
llx37, lly37 = 14,15
#8
llx18, lly18 = 14,15
llx28, lly28 = 6,7
llx38, lly38 = 12,13
#9
llx19, lly19 = 16, 17
llx29, lly29 = 18, 19
llx39, lly39 = 8, 9
#10
llx110, lly110 = 18, 19
llx210, lly210 = 10, 11
llx310, lly310 = 16, 17
#11
llx111, lly111 = 20, 21
llx211, lly211 = 16, 17
llx311, lly311 = 24, 25
#12
llx112, lly112 = 22, 23
llx212 , lly212 = 26, 27
llx312, lly312 = 28, 29
#13
llx113, lly113 = 24,25
llx213, lly213 = 26,27
llx313, lly313 = 28, 29
#14
llx114, lly114 = 26, 27
llx214 , lly214 = 22, 23
llx314, lly314 = 24, 25
#15
llx115 , lly115 = 28, 29
llx215 , lly215 = 22, 23
llx315 , lly315 = 24, 25



# Profile boundary on training data
for id1 in range(2000):
                            
        training_profile[id1, 0, :] = normal_profile_pixels(Xx_train,yy_train,id1,5,llx11,lly11,llx21,lly21,llx31,lly31)
        training_profile[id1, 1, :] = normal_profile_pixels(Xx_train,yy_train,id1,5,llx12,lly12,llx22,lly22,llx32,lly32)
        training_profile[id1, 2, :] = normal_profile_pixels(Xx_train,yy_train,id1,5,llx13,lly13,llx23,lly23,llx33,lly33)
        training_profile[id1, 3, :] = normal_profile_pixels(Xx_train,yy_train,id1,5,llx14,lly14,llx24,lly24,llx34,lly34)
        training_profile[id1, 4, :] = normal_profile_pixels(Xx_train,yy_train,id1,5,llx15,lly15,llx25,lly25,llx35,lly35)
        training_profile[id1, 5, :] = normal_profile_pixels(Xx_train,yy_train,id1,5,llx16,lly16,llx26,lly26,llx36,lly36)
        training_profile[id1, 6, :] = normal_profile_pixels(Xx_train,yy_train,id1,5,llx17,lly17,llx27,lly27,llx37,lly37)
        training_profile[id1, 7, :] = normal_profile_pixels(Xx_train,yy_train,id1,5,llx18,lly18,llx28,lly28,llx38,lly38)
        training_profile[id1, 8, :] = normal_profile_pixels(Xx_train,yy_train,id1,5,llx19,lly19,llx29,lly29,llx39,lly39)
        training_profile[id1, 9, :] = normal_profile_pixels(Xx_train,yy_train,id1,5,llx110,lly110,llx210,lly210,llx310,lly310)
        training_profile[id1, 10, :] = normal_profile_pixels(Xx_train,yy_train,id1,5,llx111,lly111,llx211,lly211,llx311,lly311)
        training_profile[id1, 11, :] = normal_profile_pixels(Xx_train,yy_train,id1,5,llx112,lly112,llx212,lly212,llx312,lly312)
        training_profile[id1, 12, :] = normal_profile_pixels(Xx_train,yy_train,id1,5,llx113,lly113,llx213,lly213,llx313,lly313)
        training_profile[id1, 13, :] = normal_profile_pixels(Xx_train,yy_train,id1,5,llx114,lly114,llx214,lly214,llx314,lly314)
        training_profile[id1, 14, :] = normal_profile_pixels(Xx_train,yy_train,id1,5,llx115,lly115,llx215,lly215,llx315,lly315)
    
training_profile = np.nan_to_num(training_profile[:, :, :])

co_points0, test_profile[0, :] = normal_profile_pixels(Xx_test,yy_test,1,15,llx11,lly11,llx21,lly21,llx31,lly31)
co_points1, test_profile[1, :] = normal_profile_pixels(Xx_test,yy_test,1,15,llx12,lly12,llx22,lly22,llx32,lly32)
co_points2,test_profile[2, :] = normal_profile_pixels(Xx_test,yy_test,1,15,llx13,lly13,llx23,lly23,llx33,lly33)
co_points3,test_profile[3, :] = normal_profile_pixels(Xx_test,yy_test,1,15,llx14,lly14,llx24,lly24,llx34,lly34)
co_points4,test_profile[4, :] = normal_profile_pixels(Xx_test,yy_test,1,15,llx15,lly15,llx25,lly25,llx35,lly35)
co_points5,test_profile[5, :] = normal_profile_pixels(Xx_test,yy_test,1,15,llx16,lly16,llx26,lly26,llx36,lly36)
co_points6,test_profile[6, :] = normal_profile_pixels(Xx_test,yy_test,1,15,llx17,lly17,llx27,lly27,llx37,lly37)
co_points7,test_profile[7, :] = normal_profile_pixels(Xx_test,yy_test,1,15,llx18,lly18,llx28,lly28,llx38,lly38)
co_points8,test_profile[8, :] = normal_profile_pixels(Xx_test,yy_test,1,15,llx19,lly19,llx29,lly29,llx39,lly39)
co_points9,test_profile[9, :] = normal_profile_pixels(Xx_test,yy_test,1,15,llx110,lly110,llx210,lly210,llx310,lly310)
co_points10,test_profile[10, :] = normal_profile_pixels(Xx_test,yy_test,1,15,llx111,lly111,llx211,lly211,llx311,lly311)
co_points11,test_profile[11, :] = normal_profile_pixels(Xx_test,yy_test,1,15,llx112,lly112,llx212,lly212,llx312,lly312)
co_points12,test_profile[12, :] = normal_profile_pixels(Xx_test,yy_test,1,15,llx113,lly113,llx213,lly213,llx313,lly313)
co_points13,test_profile[13, :] = normal_profile_pixels(Xx_test,yy_test,1,15,llx114,lly114,llx214,lly214,llx314,lly314)
co_points14,test_profile[14, :] = normal_profile_pixels(Xx_test,yy_test,1,15,llx115,lly115,llx215,lly215,llx315,lly315)





cov_training_profile = np.zeros([15,11,11])
# calculate np.diff & np.sum
#for id1 in range(2000):
for id1 in range(15):
    cov_training_profile[id1,:,:] = np.cov(training_profile[:, id1, :].T)
           
mean_training_profile = np.mean(training_profile[:,:,:],axis = 0)



#mahalanobis distance
dist1_argmin = []
for i in range(15):
    dist1=[]
    for j in range(31 - 11 +1):
        abc = cov_training_profile[i,:,:]
        dist = distance.mahalanobis(test_profile[i,j:j+11],mean_training_profile[i,:], abc)
        print(dist)
        dist1.append(dist)
    print(np.argmin(dist1))
    dist1_argmin.append(np.argmin(dist1))
    print(dist1[np.argmin(dist1)])
    
y=np.zeros([30])
y[0],y[1] =  co_points0[dist1_argmin[0]+6]
y[2],y[3] =  co_points1[dist1_argmin[1]+6]
y[4],y[5] =  co_points2[dist1_argmin[2]+6]
y[6],y[7] =  co_points3[dist1_argmin[3]+6]
y[8],y[9] =  co_points4[dist1_argmin[4]+6]
y[10],y[11] =  co_points5[dist1_argmin[5]+6]
y[12],y[13] =  co_points6[dist1_argmin[6]+6]
y[14],y[15] =  co_points7[dist1_argmin[7]+6]
y[16],y[17] =  co_points8[dist1_argmin[8]+6]
y[18],y[19] =  co_points9[dist1_argmin[9]+6]
y[20],y[21] =  co_points10[dist1_argmin[10]+6]
y[22],y[23] =  co_points11[dist1_argmin[11]+6]
y[24],y[25] =  co_points12[dist1_argmin[12]+6]
y[26],y[27] =  co_points13[dist1_argmin[13]+6]
y[28],y[29] =  co_points14[dist1_argmin[14]+6]

# Aligning the points (Step 1 for Active Shape model)
mean_shape, X_train = generalized_procrustes_analysis(Xx_train)
    
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

# Calculating covariance of (30) model ponts across 2000 samples/observations
x_cor = np.cov(X_train.T)

# Applying PCA  (Step 2 for Active Shape model)
eigval, P = np.linalg.eig(x_cor)

reverse_indices = np.argsort(eigval)   # ascending
r_ind = reverse_indices[::-1]          # descending

#eigval2 = eigval[r_ind]
P2 = P[:, r_ind]
# Take 29 principal components
P = P2[:, :29]


# Min max X, Y of an 1 text_data
box = Xx_test.iloc[1,:]
min_x = np.amin(box[::2])
min_y = np.amin(box[1::2])
max_x = np.amax(box[::2])
max_y = np.amax(box[1::2])


# Matching model points to target points
# initially the x_bar is mean value and b = 0


x_bar = mean_shape
b = np.zeros([P.shape[1], 1])
print('b: ', b.shape)
i = 0
while True:
    print(i)
    t = np.dot(P, b)
    xx = x_bar + t.reshape(-1)
    print(xx)
    #y = xx + np.random.randint(10)
    #print(y - x_bar)
    #b = np.dot(P.T,  (y - x_bar))
    if i == 1:
        xx = y - 48
    b = np.dot(P.T,  (xx - x_bar))
    print(b)
     #if the convergence did  change, break the cycle
    i += 1
     #if new != current:
    if i % 70 == 0:
        break

    
# try n run
# show a random subset of images from the dataset
box1 = yy_test[:,:,1]

plt.close('all')


shift_y, shift_x = np.array(box1.shape[:2]) / 2.

x_feature_coords = np.array(xx[::2].tolist() + shift_x)
y_feature_coords = np.array(xx[1::2].tolist() + shift_y)
plt.imshow(box1, cmap='gray');
plt.scatter(x_feature_coords,y_feature_coords,c='r',s=12)
#plt.set_axis_off()
#ax.set_title('image index = %d' %(xx),fontsize=10)


