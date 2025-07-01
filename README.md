## Introduction

This project uses an artificial neural network to try and predict a given handwritten digit.The digit is is taken as input through an IP webcam using which we can access our phone's camera on our laptop and capture a frame which we are happy with. This frame is then processed before running it through the neural network and ultimately giving a prediction

The dataset used for this network is provided and is handwritten by me. 50 samples of each digit (0-9) are used 

# description/overview:

 1. digit_regnition.py-main script (code to be executed)
 2. neural_network.py-contains our neural network
 3. frame_capture.py-to take user input ie. the frame containing handwritten digit
 4. data_processing.py-to process data in order to make it fit for training
 5. data folder-folder containing all raw data as well as its preprocessed form

# how to use

 1. In order to use the program, run the main script in the terminal. 
 2. After this you will be prompted to enter the ip which connects your phone's webcam to laptop display(ensure that your phone and laptop are connected to the same network).
 3. Ensure that ur surrounded by proper lighting along with the use of dark colour over a light background to get a clear frame for processing
 4. Once your satisfied with the frame, click 'c' to capture and it
 5. This frame will be processed and will return an output

# conclusion

I took roughly 10 days to complete this project and im satisfied given that it's my first real project.
My model's accuracy however is only ~25% and can definitely be improved in future projects with a better network and a broader data set.
