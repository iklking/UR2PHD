#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pylab as py 
#%matplotlib inline 
from IPython.display import Image, Audio
import math
import PIL 
#from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams
from matplotlib import pylab
from scipy import ndimage
import matplotlib
import wave
import sys
import random as random

"""
# For reference
# this function collects patches from black and white images
def collectPatchesBW(numPatches, patchWidth, filePath):
    maxTries = numPatches * 50
    firstPatch = 0 # the first patch number accepted from an image
    firstTry = 0 # the first attempt to take a patch from the image
    patchCount = 0 # number of collected patches
    tryCount = 0 # number of attempted collected patches
    numPixels = patchWidth * patchWidth
    patchSample = np.zeros([patchWidth,patchWidth],'double')
    patch = np.zeros([numPixels,1],'double')
    imgPatches = np.zeros([numPixels,numPatches],'double')                                          
    # chooses the image that we're sampling from
    imgCount = 1  
    image = PIL.Image.open(filePath + str(imgCount) + '.jpg')
    imageHeight, imageWidth, imageChannels = matplotlib.pyplot.imread(filePath + str(imgCount) + '.jpg').shape
    image = image.convert('L')
    image = np.asarray(image, 'double').transpose()
    # normalizing the image
    image -= image.mean()
    image /= image.std()
    while patchCount < numPatches and tryCount < numPatches:
        tryCount += 1
        if (tryCount - firstTry) > maxTries/2 or (patchCount - firstPatch) > numPatches/2:
        # change the image sampled from to the next in the folder
            imgCount += 1
            image = PIL.Image.open(filePath + str(imgCount) + '.jpg')
            imageHeight, imageWidth, imageChannels = matplotlib.pyplot.imread(filePath + str(imgCount) + '.jpg').shape
            image = image.convert('L')
            image = np.asarray(image, 'double').transpose()
            # normalizing the image
            image -= image.mean()
            image /= image.std()
            firstPatch = patchCount
            firstTry = tryCount
        #starts patch collection in a random space
        px = np.random.randint(0,imageWidth - patchWidth)
        py = np.random.randint(0,imageHeight - patchWidth)
        patchSample = image[px:px+patchWidth,py:py+patchWidth].copy()
        patchStd = patchSample.std()
        if patchStd > 0.0: # > 0 to remove blank/uninteresting patches for speed
            # create the patch vector    
            patch = np.reshape(patchSample, numPixels)     
            patch = patch - np.mean(patch)         
            imgPatches[:,patchCount] = patch.copy()
            patchCount += 1  
    return imgPatches
"""



# Helper function for concatenating images below, for BW video
# thanks to here: https://note.nkmk.me/en/python-pillow-concat-images/ Look there for my rationale
def concatenateBySimilarHeightBW(imageOne, imageTwo):
    i_1 = PIL.Image.fromarray(imageOne)
    i_2 = PIL.Image.fromarray(imageTwo)
    returnImage = PIL.Image.new('L', (i_1.width + i_2.width, i_1.height))
    returnImage.paste(i_1, (0,0))
    returnImage.paste(i_2, (i_1.width, 0))
    # print("concatenate Successful!")
    return returnImage


"""
This is our patch collection for BW video. Here is a description and my (Austin's)
thoughts about it so far.
Function parameters:
    numPatches = The number of patches we want. Same as the above.
    patchWidth = The dimensions of the image square from the video. Same as above.
    frameCount = The number of frames to be pulled from the video and for this
        patch. This is the time dimension. Only new parameter.
    filePath   = where the function can find the files.
"""
def collectPatchesVideoBW(numPatches, patchWidth, frameCount, filePath):
    # setup
    maxTries = numPatches * 50   # The multiplier here can be modified at our discretion for fine-tuning
    firstPatch = 0  # 
    firstTry = 0
    patchCount = 0
    tryCount = 0
    vectorLength = patchWidth * patchWidth * frameCount
    # patchSample = np.zeros([[patchWidth, patchWidth],frameCount], 'double')
    # patch = np.zeros([vectorLength, 1], 'double')
    vidPatches = np.zeros([vectorLength, numPatches], 'double')

    # now choose the video to be sampled from. start with the first for convenience
    vidCount = 1
    # WARNING = the below line has been hardcoded. Change this for videos with different numbers of frames.
    startingFrame = random.randint(1, 200 - frameCount)    # This will be randomly chosen from the 200 frames in each of our dataset's videos
    imageCollection = [] # This holds all the images collected between the different frames
    for x in range(frameCount):
        imageID = str(startingFrame + x) if (startingFrame + x) // 100 > 0 else \
                '0' + str(startingFrame + x) if (startingFrame + x) // 10 > 0 else \
                '00' +str(startingFrame + x) 
        pathTuple = (filePath,str(vidCount),imageID)
        absolutePath = "/".join(pathTuple)
        # print(absolutePath)
        # print( matplotlib.pyplot.imread(str(absolutePath) + '.tiff').shape)
        image = PIL.Image.open(str(absolutePath) + '.tiff')
        imageHeight, imageWidth = matplotlib.pyplot.imread(str(absolutePath) + '.tiff').shape
        image = np.asarray(image, 'double').transpose()
        # normalizing the image, as above
        image -= image.mean()
        image /= image.std()
        imageCollection.append(image)
    
    patchCollection = [] # This holds the patches derived from the images. The quantity should be equal to frameCount, one image patch per frame.
    # The above will need to be flattened out into a string for Hannah's function below. Not yet complete
    while patchCount < numPatches and tryCount < numPatches:
        tryCount += 1
        if (tryCount - firstTry) > maxTries/2 or (patchCount - firstPatch) > numPatches/2:
            # choose another video to be sampled from. Regather images from it
            startingFrame = random.randint(1, 201 - frameCount)    # This will be randomly chosen from the 200 frames in each of our dataset's videos
            imageCollection.clear() # Clears the list to refill it with new images.
            for x in range(frameCount):
                imageID = str(startingFrame + x) if (startingFrame + x) // 100 > 0 else \
                        '0' + str(startingFrame + x) if (startingFrame + x) // 10 > 0 else \
                        '00' +str(startingFrame + x) 
                pathTuple = (filePath,str(vidCount),imageID)
                absolutePath = "/".join(pathTuple)
                # print(absolutePath)
                image = PIL.Image.open(str(absolutePath) + '.tiff')
                imageHeight, imageWidth = matplotlib.pyplot.imread(str(absolutePath) + '.tiff').shape
                image = np.asarray(image, 'double').transpose()
                # normalizing the image, as above
                image -= image.mean()
                image /= image.std()
                imageCollection.append(image)
            firstPatch = patchCount
            firstTry = tryCount
        # Here starts patch collection in a random space. This starting point will hold for all frames' image patches.
        px = np.random.randint(0, imageWidth - patchWidth)
        py = np.random.randint(0, imageHeight - patchWidth)
        # Now pull the same patch from each frame
        for x in range(frameCount):
            patchSample = image[px:px+patchWidth,py:py+patchWidth].copy()
            patchStd = patchSample.std()
            # If any of the patches are statistically uninteresting, remove entire section
            if patchStd >= 8:
                patchCollection.clear()
                break
            # create the patch vector    
            # note to self: flatten patchCollection then add to vidPatches[]
            #patch = np.reshape(patchSample, numPixels)     
            #patch = patch - np.mean(patch)   HEY - y'all think this should go for each image or the concatenated one? I didn't decide tonight. only a minor point
            patchCollection.append(patchSample.copy())
        if len(patchCollection) != 0: #if there are no Patches, can't be reshaped and added to vidPatches
            # At this point patchCollection contains frameCount # of flattened image vectors. Need to append into one overall flattened int/string  
            # note to self: explain to Hannah that instead of a flattened string of a pixels these are actually concatenated Image (from PIL library) objects
                # She will need to to adjust the width by the imageWidth to get starting coordinates of each new Patch
            helperImage = patchCollection[0]
            for x in range(frameCount - 1): #So runs 3 times if frameCount is 4
                helperImage = np.array(concatenateBySimilarHeightBW(helperImage,patchCollection[x+1]))

            patch = np.reshape(helperImage, vectorLength)
            vidPatches[:, patchCount] = patch.copy()
            # vidPatches.append(patch)
            patchCollection.clear()
            patchCount += 1
    return vidPatches  


def showSlicesBW(prePatches, showPatchNum = 16, frameNum = 4, display=True):
    patches = prePatches
    # totalPatches = patches.shape[1]
    dataDim = patches.shape[0]
    frameDim = dataDim // frameNum
    patchWidth = int(np.round(np.sqrt(frameDim)))

    # extract show_patch_num patches
    displayPatch = np.zeros([dataDim, showPatchNum], float)

    for i in range(0, showPatchNum):
        # patch_i = i * totalPatches // showPatchNum
        patch_i = i
        patch = patches[:,patch_i].copy()
        pmax  = patch.max()
        pmin = patch.min()
        # fix patch range from min to max to 0 to 1
        if pmax > pmin: 
            patch = (patch - pmin) / (pmax - pmin)
        displayPatch[:,i] = patch.copy()
    obw = 5    # border width
    ibw  = 2
    pw = patchWidth
    fw = (patchWidth + ibw) * frameNum - ibw
    patchesY = int(np.sqrt(showPatchNum))
    patchesX = int(np.ceil(float(showPatchNum) / patchesY))
    patchImg = displayPatch.max() * np.ones([(fw + obw) * patchesX - obw, patchesY * (pw + obw) - obw], float)
    for i in range(0,showPatchNum):
        y_i = i // patchesY
        x_i = i % patchesY
        fullPatch = np.ones([fw, pw], float)
        for j in range(0,frameNum):
            temp = displayPatch[ (j * frameDim) :(j+1) * frameDim, i].reshape((pw,pw))
            fullPatch[ j *(pw+ibw):(j) *(pw+ibw) + pw,:] = temp.copy()

            patchImg[x_i*(fw+obw):x_i*(fw+obw)+fw, y_i*(pw+obw):y_i*(pw+obw)+pw] = fullPatch

    if display:
        py.bone()
        py.imshow(patchImg.T, interpolation='nearest')
        py.axis('off')
    return