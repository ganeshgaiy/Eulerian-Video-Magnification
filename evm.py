import cv2 as cv #type: ignore
import numpy as np #type: ignore

image = cv.imread('./test-images/face-image.jpg')
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image = cv.resize(image,(540,760))
# cv.imshow('Image', image)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Task1:convert RGB Image to YIQ color space
T_matrix = np.array([[0.299,0.59590059,0.21153661], [0.587,-0.27455667,-0.52273617], [0.114, -0.32134392, 0.31119955]])
rows, cols = image.shape[:2]

temp = np.zeros((rows, cols, 3), dtype=np.float64)

for i in range(rows):
    for j in range(cols):
        temp[i,j] = np.dot(T_matrix, image[i,j])

cv.imshow('yiq',temp)
cv.waitKey(0)
# def rgb2ntsc(src):
#     [rows,cols]=src.shape[:2]
#     dst=np.zeros((rows,cols,3),dtype=np.float64)
#     T = np.array([[0.114, 0.587, 0.298], [-0.321, -0.275, 0.596], [0.311, -0.528, 0.212]])
#     for i in range(rows):
#         for j in range(cols):
#             dst[i, j]=np.dot(T,src[i,j])
#     return dst

# cv.imshow('yiq', rgb2ntsc(image))
# cv.waitKey(0)


#1 convert the RGB video or image to YIQ color space
