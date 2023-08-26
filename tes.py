import cv2
import numpy as np
import xlsxwriter 
from collections import Counter
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops, regionprops_table
from sklearn.cluster import KMeans

image       = cv2.imread('dataset/base6.bmp')
gray        = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
mask        = cv2.dilate(thresh1.copy(),None,iterations=10)
mask        = cv2.erode(mask.copy(),None,iterations=10)
segmented   = cv2.bitwise_and(image, image, mask=mask)

contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
selected    = max(contours,key=cv2.contourArea)
x,y,w,h     = cv2.boundingRect(selected)
cropped     = segmented[y:y+h,x:x+w]
# cv2.rectangle(cropped, (x, y), (x+w, y+h), (0, 255, 0), 2)
mask        = mask[y:y+h,x:x+w]
gray        = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
# ----------------------------------------
rows,cols = cropped.size
# for i in range(0,rows):
#     for j in range(0,cols):
#         if (cropped[i,j][0]<=0):
#             cropped[i,j][3] = cv2.norm(cropped[i,j] - (255, 255, 255, 255))
# ----------------------------------------

# hsv_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
# image = hsv_image.reshape((hsv_image.shape[0] * hsv_image.shape[1], 3))
# clt = KMeans(n_clusters = 3)
# labels = clt.fit_predict(image)
# label_counts = Counter(labels)
# dom_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

# dom_color_hsv = np.full(cropped.shape, dom_color, dtype='uint8')
# dom_color_bgr = cv2.cvtColor(dom_color_hsv, cv2.COLOR_HSV2BGR)
# output_image = np.hstack((cropped, dom_color_bgr))

cv2.imshow('Image Dominant Color', cropped.size)
# print(dom_color[0])
cv2.waitKey(0)
# ----------------------------------------------------
# cv2.imshow('tes',cropped)

# label_img    = label(mask)
# props2       = regionprops(label_img)
# area         = getattr(props2[0], 'area')
# perimeter    = getattr(props2[0], 'perimeter')
# metric       = (4*np.pi*area)/(perimeter*perimeter)
# eccentricity = getattr(props2[0], 'eccentricity')
# print(perimeter)
# ----------------------------------------------------
cv2.waitKey(0)