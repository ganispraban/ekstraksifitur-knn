import cv2

jenis           = ['base','cermai','dapdap','JS','kayutoktok','MD','menuh','piduh','pucuk','pule','TB']
jum_per_data    = 25

for i in jenis:
    for j in range(1,26): 
        file_name = "dataset/" + i + str(j) + ".bmp"

        src = cv2.imread(file_name, 1)
        tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        _,alpha = cv2.threshold(tmp,127,255,cv2.THRESH_BINARY_INV)
        alpha   = cv2.dilate(alpha.copy(),None,iterations=10)
        alpha   = cv2.erode(alpha.copy(),None,iterations=10)
        b, g, r = cv2.split(src)
        rgba = [b,g,r, alpha]
        dst = cv2.merge(rgba,4)

        contours, hierarchy = cv2.findContours(alpha,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        selected    = max(contours,key=cv2.contourArea)
        x,y,w,h     = cv2.boundingRect(selected)
        cropped     = dst[y:y+h,x:x+w]

        cv2.imwrite("preprocessed/" + i + str(j) + ".png", cropped)        