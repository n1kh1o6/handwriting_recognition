import cv2 as cv
import numpy as np


# MNIST format --> (x,y) tuple (for data storage)
# x(input) is numpy array of shape(sample_size,image) where image in turn is a numpy array of shape(784,1)
# y(output) is a numpy array of shape(sample_size,1) where each entry is a label for the image (0-9)

file_name=1
folder_name=0

# 10 digits(0-9) with 50 samples each
sample_size=50*10

training_input=np.zeros((sample_size,(784,1)))
training_output=np.zeros((sample_size,(10,1)))

sample_count=0

# to give one hot encoding to each label
def vectorized(j):
    result=np.zeros((10,1))
    result[j]=1
    return result

def load_data():
    while(True):

        if sample_count==500:
            break

        if file_name==50:
            file_name=1
            folder_name+=1

        url=f"./data/{folder_name}/{file_name}.jpg"

        # Load the image in grayscale
        gray = cv.imread(url, cv.IMREAD_GRAYSCALE)

        # Binarize: converts from grayscale to black and white(pixels are either 0 or 255 where 128 is the threshold and the final argument inverts the picture making it ideal for contour detection )
        _, thresh = cv.threshold(gray, 128, 255, cv.THRESH_BINARY_INV)

        # Find contours (boundaries of white blobs in the image where the second and third arguments are used to retrieve the outermost contour,ignoring all holes inside and to simplify the contour to save memory respectively)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

        # Find the largest contour (assumed to be the digit) avoiding tiny specks and other irregularities
        # the bounding rect function finds the rectangle bounding the contour where x,y is the top-left corner and w,h are the width and height respectively
        cnt = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(cnt)

        # Crop the digit using the bounding box (removes empty space around the digit)
        digit = thresh[y:y+h, x:x+w]

        # Create a square canvas of size max_dim,max_dim filled with 0s (to ensures digit is centered within a square to preserve aspect ratio)
        max_dim = max(w, h)
        square_digit = np.zeros((max_dim, max_dim), dtype=np.uint8)

        #calculates center offsets so that digit is placed in the center of the square
        x_offset = (max_dim - w) // 2
        y_offset = (max_dim - h) // 2
        square_digit[y_offset:y_offset+h, x_offset:x_offset+w] = digit

        # Resize to 28x28
        final_img = cv.resize(square_digit, (28, 28), interpolation=cv.INTER_AREA)

        # Normalize to [0, 1] for neural network input
        normalized = final_img.astype(np.float32) / 255.0

        # reshaping it to a column vector
        np.reshape(normalized,(784,1))

        training_input(sample_count)=normalized
        training_output(sample_count)=vectorized(folder_name)

        file_name+=1
        
        sample_count+=1

    # training data is a list of tuples where each image is associated with its label vector
    training_data=zip(training_input,training_output)
    return training_data

        
