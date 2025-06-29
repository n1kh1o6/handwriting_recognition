import numpy as np
import random

# helper functions
def sigma(z):
    return 1.0/(1+np.exp(-z))

def sigma_prime(z):
    return sigma(z)*(1-sigma(z))

class Neural_Network:
    
    # nodes is a list of the number of nodes in each layer
    # weights and biases are matrices consisting of weight and bias values for the neural network initialised initially to random values from N(0,1)--> standard normal distribution
    def __init__(self,nodes):
        self.nodes=nodes
        self.layers=len(nodes)
        self.weights=[]
        self.biases=[]
        for i in range (1,self.layers):
            self.weights.append(np.random.randn(nodes[i],nodes[i-1]))# array of shape y,x
            self.biases.append(np.random.randn(nodes[i],1))# array of shape y,1
    
    #function for the formation of initial mini batches
    def SGD(self,training_data,epochs,mini_batch_size,lr):
        n=len(training_data)
        for i in range(epochs):
            mini_batches=[]
            random.shuffle(training_data)
            for k in range(0,n,mini_batch_size):
                mini_batches.append(training_data[k:k+mini_batch_size])
            for mini_batch in mini_batches:
                self.update_parameters(mini_batch,lr)
            
    def update_parameters(self,mini_batch,learning_rate):
        db=[] #db and dw aren't the actual cost derivatives, theyre just the changes that need to be added
        dw=[] #each time 
        m=len(mini_batch)
        for w,b in zip(self.weights,self.biases):
            db.append(np.zeros(b.shape))
            dw.append(np.zeros(w.shape))
        for x,y in mini_batch:
            delta_db,delta_dw=self.backprop(x,y)
            db=[db+ddb for db,ddb in zip(db,delta_db)]
            dw=[dw+ddw for dw,ddw in zip(dw,delta_dw)]
        self.weights = [w-(learning_rate/m)*nw
                        for w, nw in zip(self.weights, dw)]
        self.biases = [b-(learning_rate/m)*nb
                       for b, nb in zip(self.biases, db)]

    #to calculate delc/delw and delc/delb (gradients of the cost function for a given x) using backpropogation formulae
    def backprop(self,x,y):
        db=[] #here db and dw and the actual cost derivatives which will be calculated 
        dw=[] 
        for w,b in zip(self.weights,self.biases):
            db.append(np.zeros(b.shape))
            dw.append(np.zeros(w.shape))
        activation=x
        activations=[x]
        zs=[]
        for w,b in zip(self.weights,self.biases):
            z=np.dot(w,x)+b
            zs.append(z)
            activation=sigma(z)
            activations.append(activation)
        error=self.cost_derivative(y,activations[-1])*sigma_prime(zs[-1])
        db[-1]=error 
        dw[-1]=error@(activations[-2].T)
        for i in range(2,self.layers):
            error=((self.weights[-i+1].T)@error)*sigma_prime(zs[-i])
            db[-i]=error 
            dw[-i]=error@(activations[-i-1].T)
        return (db,dw)

    # function to return C'
    def cost_derivative(self,y,a):
        return a-y
    
    #to evaluate output based on our model for the network
    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigma(np.dot(w, a) + b)
        return a

            

        