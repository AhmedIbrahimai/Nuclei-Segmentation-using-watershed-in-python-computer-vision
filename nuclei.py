import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io
import os
os.chdir(r'E:\\')

img = cv2.imread("nu.png")
plt.imshow(img) 

#
cells=img[:,:,0]  
plt.imshow(cells) 


#pixels_to_um = 0.454 # 1 pixel = 454 nm (got this from the metadata of original image)

#Threshold image to binary using OTSU. ALl thresholded pixels will be set to 255
ret1, thresh = cv2.threshold(cells, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(thresh) #This is our image to be segmented further using watershed


# Morphological operations to remove small noise - opening
#To remove holes we can use closing
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)
plt.imshow(opening) #This is our image to be segmented further using watershed

from skimage.segmentation import clear_border
opening = clear_border(opening) #Remove edge touching grains
plt.imshow(opening, cmap='gray') #This is our image to be segmented further using watershed

#Check the total regions found before and after applying this. 

#STEP 1: Sude background 
#Now we know that the regions at the center of cells is for sure cells
#The region far away is background.
#We need to extract sure regions. For that erode a few times. 
#But we have cells touching, so erode alone will not work. 
#To separate touching objects, the best approach would be distance transform and then thresholding.

# let us start by identifying sure background area
# dilating pixes a few times increases cell boundary to background. 
# This way whatever is remaining for sure will be background. 
#The area in between sure background and foreground is our ambiguous area. 
#Watershed should find this area for us. 
sure_bg = cv2.dilate(opening,kernel,iterations=1)
plt.imshow(sure_bg, cmap='gray') #Dark region is our sure background

# Finding sure foreground area using distance transform and thresholding
#intensities of the points inside the foreground regions are changed to 
#distance their respective distances from the closest 0 value (boundary).
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
plt.imshow(dist_transform, cmap='gray') #Dist transformed img. 


#Let us threshold the dist transform by starting at 1/2 its max value.
print(dist_transform.max()) 
ret2, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)
plt.imshow(sure_fg, cmap='gray')

#Later you realize that 0.25* max value will not separate the cells well.
#High value like 0.7 will not recognize some cells. 0.5 seems to be a good compromize

# Unknown ambiguous region is nothing but bkground - foreground
sure_fg = np.uint8(sure_fg)  #Convert to uint8 from float
unknown = cv2.subtract(sure_bg,sure_fg)
plt.imshow(unknown, cmap='gray')

#Now we create a marker and label the regions inside. 
# For sure regions, both foreground and background will be labeled with positive numbers.
# Unknown regions will be labeled 0. 
#For markers let us use ConnectedComponents. 
#Connected components labeling scans an image and groups its pixels into components 
#based on pixel connectivity, i.e. all pixels in a connected component share 
#similar pixel intensity values and are in some way connected with each other. 
#Once all groups have been determined, each pixel is labeled with a graylevel 
# or a color (color labeling) according to the component it was assigned to.
ret3, markers = cv2.connectedComponents(sure_fg)
plt.imshow(markers)

#One problem rightnow is that the entire background pixels is given value 0.
#This means watershed considers this region as unknown.
#So let us add 10 to all labels so that sure background is not 0, but 10
markers = markers+10

# Now, mark the region of unknown with zero
markers[unknown==255] = 0
plt.imshow(markers, cmap='gray')   #Look at the 3 distinct regions.

#Now we are ready for watershed filling. 
markers = cv2.watershed(img,markers)
plt.imshow(markers, cmap='gray') #Dist transformed img. 

#Let us color boundaries in yellow. 
#Remember that watershed assigns boundaries a value of -1
img[markers == -1] = [255,255,255]  
plt.imshow(img) #Dist transformed img. 

#label2rgb - Return an RGB image where color-coded labels are painted over the image.
img2 = color.label2rgb(markers, bg_label=0)

plt.imshow(img2)
cv2.imshow('Overlay on original image', img)
cv2.imshow('Colored Grains', img2)
cv2.waitKey(0)

