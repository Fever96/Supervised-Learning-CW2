import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import random
import matplotlib.pyplot as plt

def generate_train_data(m,n):
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

def generate_test_data(m,n):
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

def genetate_train_data_winnow(m,n):
    random.seed(0)
    i1=0
    X=np.zeros((m,n))
    while(i1<n):
        random_data=np.random.uniform(0,1,m)
        i2=0
        while(i2<m):
            if(random_data[i2]<0.5):
                random_data[i2]=0
                i2+=1
            else:
                random_data[i2]=1
                i2+=1
        X[:,i1]=random_data.transpose()
        i1+=1

    Y=X[:,0].reshape(m,1)
    return X,Y

#最小二乘法
class lstsqr:
    def __init__(self,x,y):
        self.x=x
        self.y=y

    def train(self):
    #用伪拟实现
        pinv = np.linalg.pinv(self.x)
        w=np.dot(pinv,self.y)
        return w

    def test(self,x_test,w):
        y_test=np.sign(np.dot(x_test,w))
        return y_test


#感知机
class preceptron:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.w = np.zeros(x.shape[1])
        self.m = 0

    def sign(self,w,x):
        y=np.sign(np.dot(x,w))
        return y

    def train(self):
        length=len(self.x)
        for i in range(length):
            tmpY=self.sign(self.x[i,:],self.w)
            if(tmpY*self.y[i]<=0):
                temp=self.y[i]*self.x[i,:]
                self.w=self.w+temp
                self.m=self.m+1
        return self.w,self.m

    def test(self,x_test,w):
        y_test=np.sign(np.dot(x_test,w))
        return y_test



class winnow:
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.w=np.ones(x.shape[1])

    def predict(self,x,w):
        n=x.shape
        #print(n)
        temp=np.dot(w,x)
        y_predict=0
        if(temp<n):
            y_predict=0
        elif(temp>=n):
            y_predict=1
        return y_predict

    def train(self):
        length=len(self.x)
        for i in range(length):
            tempY=self.predict(self.x[i,:],self.w)
            if(self.y[i]!=tempY):
                self.w=self.w*pow(2,(self.y[i]-tempY)*self.x[i,:])

        return self.w

    def test(self,x_test,w):
        n=x_test.shape[1]
        length=len(x_test)
        y_test=np.dot(x_test,w)
        for i in range(length):
            if(y_test[i]<n):
                y_test[i]=0
            else:
                y_test[i]=1
        return y_test



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
            #print(self.X_train[i])
            #print(X.shape)
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            knn_list.append((dist, self.y_train[i]))

        #对数据集剩下的部分遍历
        #更新距离
        for i in range(self.n, len(self.X_train)):
            #print(i)
            max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            #print(knn_list[max_index][0])
            #print("!!!")
            #print(dist)
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist, self.y_train[i])

        # 统计
        knn = [k[-1] for k in knn_list]
        #print(knn)
        count_pairs = knn
        max_count = sorted(count_pairs, key=lambda x: x)[-1]
        return max_count

    def estimate(self,X):
        length=X.shape[0]
        y_predict=np.ones(length)
        for i in range(length):
            y_predict[i]=self.predict(X[i,:])
        return y_predict



    # def score(self, X_test, y_test):
    #     right_count = 0
    #     n = 10
    #     for X, y in zip(X_test, y_test):
    #         label = self.predict(X)
    #         if label == y:
    #             right_count += 1
    #     return right_count / len(X_test)

def compute_sample_complexity(y_predict,y_true,m):
    count=0
    y_true=y_true.flatten()
    length1=len(y_predict)
    length2=len(y_true)
    for i in range(length2):
        if(y_predict[i]!=y_true[i]):
            count+=1
    res=count/m
    return res

def compute_sample_complexity2(y_predict,y_true):
    count=0
    y_true=y_true.flatten()
    n=len(y_true)
    for i in range(n):
        if(y_predict[i]!=y_true[i]):
            count+=1
    res=pow(2,-n)*count
    return res

def preceptron_sample_complexity():
    f="preceptron.txt"
    file=open(f,"w")
    n=1
    num_test=10000
    while(n<=100):
        m=1
        while(m<=1000):
            x,y=generate_train_data(m,n)
            x_test,y_test=generate_train_data(num_test,n)
            myPreceptron=preceptron(x,y)
            w,m1=myPreceptron.train()
            y_predict=myPreceptron.test(x_test,w)
            #print(y_predict)
            res=compute_sample_complexity(y_predict,y_true=y_test,m=num_test)
            print("m:"+str(m))
            if(res<0.1):
                file.writelines("n:"+str(n)+" m:"+str(m)+"\n")
                break
            else:
                m+=1
        print("n"+str(n))
        n=n+1

    file.close()
    draw_picture(f)

def lstsqr_sample_complexity():
    f="lstsqr.txt"
    file=open(f,"w")
    n=1
    num_test=50
    while(n<=100):
        m=1
        while(m<=1000):
            x,y=generate_train_data(m,n)
            x_test,y_test=generate_train_data(num_test,n)
            mylstsqr=lstsqr(x,y)
            w=mylstsqr.train()
            y_predict=mylstsqr.test(x_test,w)
            #print(y_predict)
            res=compute_sample_complexity(y_predict,y_true=y_test,m=num_test)
            #print(res)
            print("m:"+str(m))
            if(res<0.1):
                file.writelines("n:"+str(n)+" m:"+str(m)+"\n")
                break
            else:
                m+=1
        print("n"+str(n))
        n=n+1

    file.close()
    draw_picture(f)

def winnow_sample_complexity():
    f="winnow.txt"
    file=open(f,"w")
    n=1
    num_test=10000
    while(n<=100):
        m=1
        while(m<=1000):
            x,y=genetate_train_data_winnow(m,n)
            x_test,y_test=genetate_train_data_winnow(num_test,n)
            mywinnow=winnow(x,y)
            w=mywinnow.train()
            y_predict=mywinnow.test(x_test,w)
            #print(y_predict)
            res=compute_sample_complexity(y_predict,y_true=y_test,m=num_test)
            #print(res)
            print("m:"+str(m))
            if(res<0.1):
                file.writelines("n:"+str(n)+" m:"+str(m)+"\n")
                break
            else:
                m+=1
        print("n"+str(n))
        n=n+1

    file.close()
    draw_picture(f)

def onenn_sample_complexity():
    num_test=7 #7是最好的解
    f="onenn"+str(num_test)+".txt"
    file=open(f,"w")
    n=1
    while(n<=100):
        m=1
        while(m<=100):
            x, y = genetate_train_data_winnow(m, n)
            x_test, y_test = genetate_train_data_winnow(num_test, n)
            #print(y_test)
            myKNN = KNN(x, y)
            y_predict = myKNN.estimate(x_test)
            #print(y_predict)
            res=compute_sample_complexity(y_predict,y_true=y_test,m=num_test)
            #print(res)
            #print("m:"+str(m))
            if(res<0.1):
                print(res)
                file.writelines("n:"+str(n)+" m:"+str(m)+"\n")
                break
            else:
                m+=1
        #print("n"+str(n))
        n=n+1

    file.close()
    draw_picture(f)
    #file.close()

def draw_picture(f):
    file=open(f,"r")
    x=np.zeros(99)
    y=np.zeros(99)
    i=1
    while(i<100):
        temp1=file.readline().replace('\n','')
        temp2=temp1.split(' ')
        x[i-1]=temp2[0].split(':')[1]
        y[i-1]=temp2[1].split(":")[1]
        i=i+1
    file.close()
    plt.figure(1)
    plt.plot(x,y)
    plt.show()

#test second method to compute sample complexity
def lstsqr_sample_complexity2():
    f="lstsqr.txt"
    file=open(f,"w")
    n=1
    num_test=100000
    while(n<=100):
        m=1
        while(m<=10000000):
            x,y=generate_train_data(m,n)
            x_test,y_test=generate_train_data(num_test,n)
            mylstsqr=lstsqr(x,y)
            w=mylstsqr.train()
            y_predict=mylstsqr.test(x_test,w)
            #print(y_predict)
            res=compute_sample_complexity2(y_predict,y_true=y_test)
            print(res)
            print("m:"+str(m))
            if(res<0.1):
                file.writelines("n:"+str(n)+" m:"+str(m)+"\n")
                break
            else:
                m+=1
        print("n"+str(n))
        n=n+1

    file.close()
    draw_picture(f)
if __name__ == '__main__':
    #lstsqr_sample_complexity()
    #preceptron_sample_complexity()
    #f='preceptron_50.txt'
    #draw_picture(f)
    #winnow_sample_complexity()
    #onenn_sample_complexity()
    lstsqr_sample_complexity2()

