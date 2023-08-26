import cv2

jenis           = ['base','cermai','dapdap','JS','kayutoktok','MD','menuh','piduh','pucuk','pule','TB']
jum_per_data    = 25

file_name = "dataset/base6.bmp"

src = cv2.imread(file_name, 1)
tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
_,mask = cv2.threshold(tmp,127,255,cv2.THRESH_BINARY_INV)
mask   = cv2.dilate(mask.copy(),None,iterations=10)
mask   = cv2.erode(mask.copy(),None,iterations=10)
b, g, r = cv2.split(src)
rgba = [b,g,r, mask]
dst = cv2.merge(rgba,4)

contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
selected    = max(contours,key=cv2.contourArea)
x,y,w,h     = cv2.boundingRect(selected)
cropped     = dst[y:y+h,x:x+w]
mask        = mask[y:y+h,x:x+w]
gray        = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

# output_image = np.hstack((cropped, dom_color_bgr))
cv2.imshow('Image Dominant Color', mask)

# cv2.imwrite("test.png", cropped)
cv2.waitKey(0)