import numpy as np
import random
from collections import Counter

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

class preceptron2:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.w = np.zeros(x.shape[1])
        self.m = 0

    def sign(self,w,x):
        y=np.dot(x,w)
        return int(y)

    def train(self):
        length=len(self.x)
        for i in range(length):
            tmpY=self.sign(self.x[i,:],self.w)
            #print(tmpY)
            #print(y[i])
            if(tmpY*self.y[i]<=0):
                #print(x[i,:])
                temp=self.y[i]*self.x[i,:]
                #print(temp)
                self.w=self.w+temp
                #print(self.w)
                self.m=self.m+1
        return self.w,self.m

class winnow:
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.w=np.ones(x.shape[1])

    def prefict(self,x,w):
        n=x.shape
        #print(n)
        temp=np.dot(w,x)
        y=0
        if(temp<n):
            y=0
        elif(temp>=n):
            y=1
        return y

    def train(self):
        length=len(self.x)
        for i in range(length):
            tempY=self.prefict(self.x[i,:],self.w)
            if(y[i]!=tempY):
                self.w=self.w*pow(2,(y[i]-tempY)*x[i,:])

        return self.w


class KNN:
    def __init__(self, X_train, y_train, n_neighbors=1, p=2):
        """
        parameter: n_neighbors 临近点个数
        parameter: p 距离度量
        """
        self.n = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        # 取出n个点
        knn_list = []
        for i in range(self.n):
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            knn_list.append((dist, self.y_train[i]))

        for i in range(self.n, len(self.X_train)):
            max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist, self.y_train[i])

        # 统计
        knn = [k[-1] for k in knn_list]
        count_pairs = Counter(knn)
        max_count = sorted(count_pairs, key=lambda x: x)[-1]
        return max_count

    def score(self, X_test, y_test):
        right_count = 0
        n = 10
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right_count += 1
        return right_count / len(X_test)

if __name__ == '__main__':
    x,y=generate_data(30,100)
    myknn=KNN(x,y)
    x_test=np.ones(100)
    y_test=myknn.predict(x_test)
    print(y_test)
