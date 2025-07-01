import frame_capture
import neural_network
import data_processing
import numpy as np
import mnist_loader

ip=input("enter your webcam ip: ")
url=f"http://{ip}:8080/video"

training_data=data_processing.load_data()
frame=frame_capture.frame_capture(url)
if frame is None:
    print("Failed to capture test data. Exiting.")
    exit()

n1=neural_network.Network([784,30,10])

"""
epochs=76
mini batch size=5(total batch = 500)
learning rate=0.3
"""
n1.SGD(training_data,75,5,0.3)

prediction=n1.feedforward(frame)
max_index=np.argmax(prediction)
print("model predicts that your digit is ",max_index)