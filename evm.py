import cv2 as cv #type: ignore
import numpy as np #type: ignore

image = cv.imread('./test-images/monkey.jpeg')
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image = cv.resize(image,(540,760))
# cv.imshow('Image', image)
# cv.waitKey(0)
# cv.destroyAllWindows()

#YIQ transformation matrix
rgb_to_yiq_matrix = np.array([
    [0.299,  0.587,  0.114],
    [0.596, -0.275, -0.321],
    [0.212, -0.523,  0.311]
])

yiq_to_rgb_matrix = np.array([
        [1.0,  0.956,  0.621],
        [1.0, -0.272, -0.647],
        [1.0, -1.106,  1.703]
    ])

# Todo: simplify code 1 & 1.1
#1 convert the RGB video or image to YIQ color space
def rgbtoyiq(rgb_image):
    reshaped_img = rgb_image.reshape((-1, 3))
    yiq_flat_image = np.dot(reshaped_img, rgb_to_yiq_matrix.T)
    yiq_image = yiq_flat_image.reshape(rgb_image.shape)
    return yiq_image

#1.1 convert the YIQ video or image to RGB color space
def yiqtorbg(yiq_image):
    reshaped_img = yiq_image.reshape((-1,3))
    rgb_flat_image = np.dot(reshaped_img, yiq_to_rgb_matrix.T)
    rgb_img = rgb_flat_image.reshape(yiq_image.shape)
    rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)
    return rgb_img

yiq_img = rgbtoyiq(image)
cv.imshow('image',yiq_img)
cv.waitKey(0)

cv.imshow('image2', yiqtorbg(yiq_img))
cv.waitKey(0)




