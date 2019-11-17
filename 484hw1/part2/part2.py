import numpy as np
import cv2
import sys
import scipy.misc
from PIL import Image
from scipy import ndimage
#path of the image
path = r'ct.png'

#convert gray during input
img = cv2.imread( path,0)
size = 512
resized = cv2.resize(img, (size, size), cv2.INTER_LINEAR)
# thresholding
ret, bw_img = cv2.threshold(resized,130,255,cv2.THRESH_BINARY)
cv2.imshow("Binary Image",bw_img)
cv2.imwrite('thresh.png',bw_img)

###########################################333
# 			DILATION FROM 1ST PART
#############################################3
def dilation( source_image, struct_el, originRow, originCol):
	#number of rows and columns of the struct_el
	numrows = len(struct_el)    
	numcols = len(struct_el[0])  
	#result array for image
	resultImage = np.zeros((size,size))
	#resultImage = np.logical_not(resultImage).astype(int)
	#trace all the pixels
	for i in range (size):
		for j in range (size):
			if( source_image[i][j] == 1) : # if seas black pixel
				if( (i> originRow and j> originCol and i< size-originRow and j <size-originCol)): #check boundaries
					orShape = np.bitwise_or(source_image[ (i-originRow):(i+numrows-originRow), (j-originCol):(j+numcols-originCol) ], struct_el) #anding the pixels 
					for x in range(numrows):
						for y in range( numcols):
							resultImage[i-originRow+x][j-originCol+y] = orShape[x][y]# changing the pixels
	#change pixel to create binary image
	for i in range (size):
		for j in range (size):
		
			if( resultImage[i][j] == 1) :
				 resultImage[i][j] = 255
	#cv2.imshow("Binary Image2",source_image)
	#from array to binary
	img2 = Image.fromarray(resultImage.astype('uint8'))
	img2.save('dilation.png')
	#img2.show()
	return img2,resultImage
###########################################################3
#				EROSION FROM 1ST PART
###########################################################3333
def erosion( source_image, struct_el, originRow, originCol):
	#most part are same as dilation but only not and operator or operator
	numrows = len(struct_el)   
	numcols = len(struct_el[0]) 
	resultImage2 = np.zeros((size,size))
	#resultImage2 = np.logical_not(resultImage2).astype(int)
	for i in range (size):
		for j in range (size):
			if( source_image[i][j] == 1) :
				if( (i> originRow and j> originCol and i< size-originRow and j <size-originCol)):
					orShape = np.bitwise_and(source_image[ (i-originRow):(i+numrows-originRow), (j-originCol):(j+numcols-originCol) ], struct_el)
					if( np.array_equal( orShape , struct_el)):
						resultImage2[i][j] = 1# changing the pixels

	for i in range (size):
		for j in range (size):
			if( resultImage2[i][j] == 1) :
				 resultImage2[i][j] = 255
	#cv2.imshow("Binary Image2",source_image)
	
	img3 = Image.fromarray(resultImage2.astype('uint8'))
	if( originRow ==1):
		img3.save('erosion1.png')
	else:
		img3.save('erosion.png')
	img3.show()
	return img3,resultImage2
	
#convert to 1 0's
#https://stackoverflow.com/questions/45351402/convert-an-image-array-to-a-binarized-image
threshold = 0.2
new_indice = np.where(bw_img/255>=threshold, 1, 0)

struct_element3 = [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]
struct_element4 = [[1,1,1],[1,1,1],[1,1,1]]
struct_element5 =[[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1]]
originrow = 1
origincol = 1
originrow2 = 2
origincol2 = 2
originrow3 = 3
origincol3 = 3


############################################
#		STEPS
#############################################33
#use erosion
e_img, e_result = erosion(new_indice, struct_element4, originrow, origincol)
#dilation
e_img = cv2.imread( 'erosion.png',0)
ret, bw_img = cv2.threshold(e_img,130,255,cv2.THRESH_BINARY)
ero = np.where(bw_img/255>=threshold, 1, 0)
#e_img1, e_result2 = erosion(ero, struct_element, originrow, origincol)
#ero2 = np.where(erosion_result2/255>=threshold, 1, 0)
dimg, dresult = dilation(ero, struct_element3, originrow2, origincol2)


#gaussian
#src = cv2.imread('erosion.png', cv2.IMREAD_UNCHANGED)
#gaus = cv2.GaussianBlur(src,(5,5),cv2.BORDER_DEFAULT)
#cv2.imshow("gaus", gaus)

##############################################3333
#                 labeling
#########################################3
img = cv2.imread('dilation.png', 0)
img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  
ret, labels = cv2.connectedComponents(img)

#https://stackoverflow.com/questions/46441893/connected-component-labeling-in-python
#this site has been used to coloring the components
def color_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv2.imshow('labeled.png', labeled_img)
    cv2.imwrite('labeled.png', labeled_img)
    cv2.waitKey()

color_components(labels)

cv2.waitKey(0)
cv2.destroyAllWindows()
