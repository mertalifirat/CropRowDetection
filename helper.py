import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pyrsistent import s
import sklearn
import skimage
from skimage import filters
from skimage.measure import label
import cv2
import math
import pdb
import math
import sys



###########################################################################################
###########################################################################################
#########                                                                         #########
#########          This helper.py file consists lots of helper functions          #########
#########              that aims to solve the problems defined by paper           #########
#########                                                                         #########
###########################################################################################
###########################################################################################



### First method Identification of Grenness
def identifyGreen(img_path,resizeShape=(320,240)):
    image=Image.open(img_path)
    image=np.array(image)
    image=np.dot(image[...,:3], [1.262, -0.884, -0.311]) ### applying the formula that paper defines
    image = cv2.resize(image, dsize=resizeShape, interpolation=cv2.INTER_CUBIC) ### resizing 
    return image


### Second method Grey to Binary
def greyToBin(image):
    ### converting gray scale image to binary image by thresholding mentioned in paper
    blurred_image = skimage.filters.gaussian(image, sigma=1.0)
    t = skimage.filters.threshold_otsu(blurred_image)
    return (blurred_image < t)


### Third method Removing Smaller Objects (Removing Noise)
def remNoise(image):
    ### applying the pseudocode at the section III.
    labelledImg=label(image) ### take the image with labelled connected components
    nonzero=np.count_nonzero(image) ### count the nonzero values (whites) 
    meanVal=nonzero/np.max(labelledImg) ### computing the mean value as paper defines
    for i in range(1,np.max(labelledImg)+1): ### for each label value detect the noise and make these labelled pixels 0 (black)
        temp=np.where(labelledImg==i)
        x_coor=temp[0]
        y_coor=temp[1]
        binObj=labelledImg[temp]
        ai=len(binObj)
        if ai < meanVal/2:
            for x,y in zip(x_coor,y_coor):
                image[x][y]=0
    return image




##########################################################
####             JOINING ROW OBJECTS                 #####
##########################################################
##  Fourth method consists lots of functions as below   ##
##########################################################

### after removing the smaller objects, it is time for the joining the row objects

### first we will figure out the points of the binary object. Afterwards, the side of the binary object wrt. points,
### and the centre of the image, the computing the size of the triangle matrix that will be created for detecting the other objects.

### at the end the joining method will be executing.
def sideDetection(centre,c1,c2,p1,p2,p3,p4): 
    ### detect the where binary object is located left or right 
    ### return 0 ~ left 
    ### return 1 ~ right
    ### naive implementation for now (same structure as the paper defines)
    if (c1 < centre and c2 < centre):
        return 0
    elif (c1 > centre and c2 > centre):
        return 1
    else:
        if (p1-p2>0 and p3-p4>0):
            return 0
        elif (p1-p2<0 and p3-p4<0):
            return 1
        else:
            if (np.abs(p1-p2)>np.abs(p3-p4)):
                return 0
            elif (np.abs(p1-p2)<=np.abs(p3-p4)):
                return 1

def sizeDetection(r1,smin,smax,X):
    ### detecting the size of the triangle matrix that will be created
    return math.ceil(smin+(smax-smin)/(X-1)*(r1-1))
def triangeMatrix(side,size):
    ### creating triangle matrix wrt. side of the binary object
    res=np.ones((size,size))
    if (side): ### if it is right sided
        return np.triu(res)
    else:  ### else left sided
        for i in range(size):
            for j in range(size):
                if (i+j-1>size-2): res[i][j]=0
        return res
def determinePoints(labelledImg,i):
    temp=np.where(labelledImg==i)
    x_coor=temp[0]  ### coordinates of the binary object
    y_coor=temp[1]
    c1=np.min(y_coor)   ### c1, c2, r1, and r2 points of that binary object
    c2=np.max(y_coor)
    r1=np.min(x_coor)
    r2=np.max(x_coor) 
    p1=r1,y_coor[0]   ### p1, p2, p3, and p4 points of that binary object
    p3=r1,y_coor[np.max(np.where(x_coor==r1))]
    p2=r2,y_coor[np.where(x_coor==r2)[0][0]]
    p4=r2,y_coor[np.max(np.where(x_coor==r2))]
    return {'temp':temp,
            'x_coor':x_coor,
            'y_coor':y_coor,
            'c1':c1,'c2':c2,'r1':r1,'r2':r2,
            'p1':p1,'p2':p2,'p3':p3,'p4':p4,
            'i':i}  ### dict object

### wrap it all in one main function
def forthMain(image,X,centre):
    image=image.astype(np.uint8)
    labelledImg=label(image) ### take the image with labelled connected components (binary objects)
    for i in range(1,np.max(labelledImg)+1): ### that loop detects each binary object
        items=determinePoints(labelledImg,i)
        side=sideDetection(centre,items['c1'],items['c2'],items['p1'][1],items['p2'][1],items['p3'][1],items['p4'][1])  ### side detection which is defined above
        size=sizeDetection(items['r1'],2,0.5*X,X)  ### size detection which is defined above
        matrix=triangeMatrix(side,size)  ### creating the triangle matrix as paper defines
        if items['r1']>X/2:
            if (not side): ### left ### this might cause problem since overlapping is not well defined in paper
                tempObj=labelledImg[items['r1']-size+1:items['r1']+1,items['p3'][1]-1:items['p3'][1]+size-1]
                x=tempObj*matrix
                tempObj=np.logical_or(tempObj,matrix)
                if len(np.unique(x))>2:
                    least=sys.maxsize ### temp max value
                    poi=-1
                    for obj in np.unique(x)[1:]:
                        if (int(obj) != i):
                            itemsObj=determinePoints(labelledImg,obj)
                            distance=math.sqrt((itemsObj['p4'][0]-items['p3'][0])**2+(itemsObj['p4'][1]-items['p3'][1])**2)
                            if distance < least: 
                                least=distance
                                poi=itemsObj
                    start=int((poi['p2'][1]+poi['p4'][1])/2),poi['r2']
                    end=int((items['p1'][1]+items['p3'][1])/2),items['r1']
                    image=cv2.line(image,start,end,color=1,thickness=5)

            else: ### right ### this might cause problem since overlapping is not well defined in paper
                tempObj=labelledImg[items['r1']-size+1:items['r1']+1,items['p1'][1]-size+1:items['p1'][1]+1]
                x=tempObj*matrix
                tempObj=np.logical_or(tempObj,matrix)
                if len(np.unique(x))>2:
                    least=sys.maxsize   ### temp max value
                    poi=-1
                    for obj in np.unique(x)[1:]:
                        if (int(obj) != i):
                            itemsObj=determinePoints(labelledImg,obj)
                            distance=math.sqrt((itemsObj['p2'][0]-items['p1'][0])**2+(itemsObj['p2'][1]-items['p1'][1])**2)
                            if distance < least: 
                                least=distance
                                poi=itemsObj
                    start=int((poi['p2'][1]+poi['p4'][1])/2),poi['r2']
                    end=int((items['p1'][1]+items['p3'][1])/2),items['r1']
                    image=cv2.line(image,start,end,color=1,thickness=5)
    return image


##########################################################
####            EXTENDING LONGER OBJECTS             #####
##########################################################
##   Fifth method consists lots of functions as below   ##
##########################################################
## This method will check the incomplete parts of the row objects
## Then extending the empty part with the intuition as paper defines
## This method consists 
### Completeness check from paper
def completeness(items,X,Y):
    if ((items['r1']==0 and items['r2']==X-1 ) or 
        (items['r1']==0 and items['c1']==0 ) or
        (items['r1']==0 and items['c2']==Y-1 )):
        items['Complete']=True
    else:
        items['Complete']=False
    return items
### calculation of the U vector as paper defines
def calU(items):
    res=np.zeros(items['r2']-items['r1']+1)
    #p3=items['p3'][1]
    for j in range(items['r2']-items['r1']+1):
        p1j=items['y_coor'][np.where(items['x_coor']==items['r1']+j)[0][0]]
        p3j=items['y_coor'][np.max(np.where(items['x_coor']==items['r1']+j))]
        res[j]=math.ceil((p1j+p3j)/2.0)
    return res
### wrap it all in one main function
def fifthmain(image,rate):
    extlongImg=image[int(image.shape[0]*rate):,:] ### taking some portion of rows
    labelledImg=label(extlongImg)
    X=labelledImg.shape[0]
    for i in range(1,np.max(labelledImg)+1):
        items=determinePoints(labelledImg,i)
        items=completeness(items,labelledImg.shape[0],labelledImg.shape[1])
        if (not items['Complete']): ### If not complete
            if(items['r2']-items['r1']+1>0.7*X):  ### If the object is greater than the 0.7 of the row count
                gama=np.arange(items['r1'],items['r2']+1)
                U=calU(items)
                a,b=np.polyfit(gama,U,1)
                if (items['r1']!=0):
                    ### 15) a
                    cUpper=a*np.arange(items['r1']) + b
                    ### 16) a,b,c
                    extlongImg[np.arange(items['r1']),cUpper.astype(int)]=1
                    extlongImg[np.arange(items['r1']),cUpper.astype(int)-1]=1
                    extlongImg[np.arange(items['r1']),cUpper.astype(int)+1]=1
                if (items['r2']!=X-1):
                    ### 15) b
                    cLower=a*np.arange(items['r2'],X) +b
                    ### 17) a,b,c
                    extlongImg[np.arange(items['r2'],X).astype(int),cLower.astype(int)]=1
                    extlongImg[np.arange(items['r2'],X).astype(int),cLower.astype(int)-1]=1
                    extlongImg[np.arange(items['r2'],X).astype(int),cLower.astype(int)+1]=1
    return extlongImg

##########################################################
####          EXTENDING OTHER ROW OBJECTS            #####
##########################################################
##   Sixth method consists lots of functions as below   ##
##########################################################

###  This method's aim is find the identify the left uncomplete objects whether they are
###  part of some full row object or not
def sixthmain(image):
    return image


######################################################
#####         Extracting the row lines           #####
######################################################
## Lines can be extracted with two methods:
## i) Using Hough Transform
## ii) Polyfit with degree one
def houghRowLines():
    pass

def polyfitLines(image):
    X=image.shape[0]
    lineImg=np.copy(image)
    result=np.zeros((lineImg.shape[0],lineImg.shape[1]))
    labelld=label(image)
    plt.imshow(labelld)
    for i in range(1,np.max(labelld)+1):
        items=determinePoints(labelld,i)
        items=completeness(items,X,image.shape[1])
        if (items['Complete']):
            gama=np.arange(items['r1'],items['r2']+1)
            U=calU(items)
            a,b=np.polyfit(gama,U,1)
            cGen=a*np.arange(int(X)) + b
            for j in range(int(X)): ### once vectorized but need the below if statement
                if (0<cGen[j]<lineImg.shape[1]-1):
                    result[j,(cGen.astype(int))[j]]=1
                    result[j,(cGen.astype(int))[j]-1]=1
                    result[j,(cGen.astype(int))[j]+1]=1

    return result
