import frame_capture
import neural_network
import data_processing
import numpy as np

url=input("enter url of ip webcam site")

training_data=data_processing.load_data()
test_data=frame_capture.frame_capture(url)

n1=neural_network.Neural_Network([784,30,10])

"""
epochs=50
mini batch size=4 (total batch = 500)
learning rate=1.0
"""
n1.SGD(training_data,50,4,1.0)

prediction=n1.feedforward(test_data)
max_index=np.argmax(prediction)
print("model predicts that your digit is ",max_index)