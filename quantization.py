import cv2
import numpy as np
import matplotlib.pyplot as plt  
img = cv2.imread('6.jpeg')


def quant(img,k):
    data = np.float32(img).reshape(-1,3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
    ret,label,center = cv2.kmeans(data, k , None, criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result


plt.imshow(quant(img,5))
plt.show()
#print(type(color_3))



