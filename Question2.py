import numpy as np
import random
from scipy.optimize import leastsq

def generate_data(m,n):
    random.seed(0)
    i1=0
    X=np.zeros((m,n))
    while(i1<n):
        random_data=np.random.uniform(0,1,m)
        i2=0
        while(i2<m):
            if(random_data[i2]<0.5):
                random_data[i2]=-1
                i2+=1
            else:
                random_data[i2]=1
                i2+=1
        X[:,i1]=random_data.transpose()
        i1+=1

    Y=X[:,0].reshape(m,1)
    return X,Y

#最小二乘法
def lstsqr(x, y):
    #用伪拟实现
    pinv = np.linalg.pinv(x)
    w=np.dot(pinv,y)
    return w

#感知机
class perceptron:
    def __init__(self,x,y,a=1):
        self.x = x
        self.y = y
        self.w = np.zeros((x.shape[1],1))
        self.b = 0
        self.a = 1
    def sign(self,w,b,x):
        result = 0
        y = np.dot(x,w)+b
        return int(y)
    def train(self):
        flag = True
        length = len(self.x)
        while flag:
            count = 0
            for i in range(length):
                tmpY = self.sign(self.w,self.b,self.x[i,:])
                if tmpY*self.y[i]<=0:
                    tmp = self.y[i]*self.a*self.x[i,:]
                    tmp = tmp.reshape(self.w.shape)
                    self.w = tmp +self.w
                    self.b = self.b + self.y[i]
                    count +=1
            if count == 0:
                flag = False
        return self.w,self.b


if __name__ == '__main__':
    x,y=generate_data(100,30)
    myPerceptron=perceptron(x,y)
    weight,bias=myPerceptron.train()
    print(weight)
    print(bias)