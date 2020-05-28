import scipy
import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
import cv2
import scipy.misc

def load_data():
    train_data=h5py.File('C:\\Users\\Siddharth\\Documents\\Coursera\\train_catvnoncat.h5',"r")
    train_set_x_orig = np.array(train_data["train_set_x"][:]) # your train set features
    train_set_y_orig= np.array(train_data["train_set_y"][:])
    test_data=h5py.File('C:\\Users\\Siddharth\\Documents\\Coursera\\test_catvnoncat.h5',"r")
    test_set_x_orig=np.array(test_data["test_set_x"][:])
    test_set_y_orig=np.array(test_data["test_set_y"][:])
    print(train_set_x_orig.shape)
    return (train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig)
    # print(list(train_data.keys()))


def initialize_parameter(dim):
    w=np.zeros((dim,1))
    b=0
    return w,b

def forward_propagation(w,b,x,y):
    z=np.dot(w.T,x)+b

    a=1/(1+np.exp(-z))
    m=x.shape[1]
    cost=(-1/m)*np.sum(np.dot(y,np.log(a).T)+np.dot(1-y,np.log(1-a).T))

    dW=(1/m)*np.dot(x,(a-y).T)
    db=(1/m)*np.sum((a-y),axis=1)

    return dW,db,cost

def predict(w,b,x):
    y_pred=np.zeros((1,x.shape[0]))
    a=np.dot(w.T,x)+b
    a=1/(1+np.exp(-a))
    y_pred=a>0.5
    return y_pred
def optimize(w,b,x,y,learning_rate,iteration):
    for i in range(iteration):
        dw,db,cost=forward_propagation(w,b,x,y)
        if i%100==0:
            print(str(i)+" "+str(cost))
        w=w-learning_rate*dw
        b=b-learning_rate*db
    return w,b


train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig=load_data()

train_set_x=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
train_set_x=train_set_x/255
test_set_x=test_set_x/255

print(train_set_x.shape,test_set_x.shape)

w,b=initialize_parameter(12288)

w,b=optimize(w,b,train_set_x,train_set_y_orig,0.005,2000)

cap=cv2.VideoCapture(0)
window_name='sd'
while True:
    _,img=cap.read()
    
##    img=cv2.imread('Capture.jpg')
    
    
##    arrimg=np.array(img)
##    arrimg=arrimg
    font = cv2.FONT_HERSHEY_SIMPLEX 
  
# org 
    org = (50, 50) 
      
    # fontScale 
    fontScale = 1
       
    # Blue color in BGR 
    color = (255, 0, 0) 
      
    # Line thickness of 2 px 
    thickness = 2
       
    # Using cv2.putText() method 
    
    img2=cv2.resize(img,(64,64))
    im=img2.flatten().reshape(1,12288).T  
    y=predict(w,b,im)
    if (y==True):
        image = cv2.putText(img, 'Cat', org, font,  
                       fontScale, color, thickness, cv2.LINE_AA)
    else:
        image = cv2.putText(img, 'NonCat', org, font,  
                       fontScale, color, thickness, cv2.LINE_AA)
        
    cv2.imshow(window_name,image)

##    cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('w'):
        break
    
    
    






    





