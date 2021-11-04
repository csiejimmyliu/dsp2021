

import numpy as np
import os
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt


def draw(x, name):
    plt.ioff()
    plt.imshow(x.reshape(28, 28), 'gray')
    plt.title(name)
    plt.axis('off')


def project(a, b):
    #project a on b space
    s = np.matmul(b.T, b)
    invs = np.linalg.pinv(s)
    re = b@invs@b.T@a
    return re


def cpca(x):
    #centered pca
    length, size = x.shape

    mean = np.zeros(size)
    for i in range(length):
        mean += x[i]
    mean /= length

    for i in range(length):
        x[i] -= mean

    s = np.matmul(x.T, x)
    s /= length-1
    w, v = np.linalg.eig(s)

    v = v.real
    v = v.T

    for i in range(size):
        if w[i] < 0:
            w[i] *= -1
            v[i] *= -1

    m = np.argsort(-1*w)

    return mean, x, w, v, m


def omp(data, orix, k):
    #omp, return list of index of basis
    x = orix*1
    d = data*1
    use = [True]*len(d)
    re = []
    for i in range(len(d)):
        d[i] /= (np.inner(d[i], d[i])**0.5)

    b = []
    r = x*1

    for turn in range(k):
        maximum = float('-inf')
        index = 0
        for i in range(len(d)):
            if use[i] == True and abs(np.inner(d[i], r)) > maximum:
                maximum = abs(np.inner(d[i], r))
                index = i
        re.append(index)
        use[index] = False
        b.append(d[index])
        B = np.array(b).T
        r = x-project(x, B)
    return np.array(re)


mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
target = mnist['target']
length, vecsize = data.shape



os.makedirs('./fig')

#Q1
mean = np.zeros(vecsize)

for i in range(length):
    mean += data[i]
mean /= length

plt.imshow(mean.reshape(28, 28), 'gray')
plt.title('mean')
plt.axis('off')
plt.savefig('./fig/Q1.jpg')

#Q2
vec5 = []

for i in range(length):
    if target[i] == '5':
        vec5.append(data[i])

vec5 = np.array(vec5)

mean, cvec, evalue, evec, order = cpca(vec5)
plt.clf()
fig = plt.figure(figsize=(9, 3))
fig.patch.set_facecolor('white')
for i in range(3):
    plt.subplot(131+i)
    draw(evec[order[i]], 'λ='+str(int(evalue[order[i]])))

plt.savefig('./fig/Q2.jpg')


#Q3
bnum = [3, 10, 30, 100]
out = []
first5 = vec5[0]

out.append(first5)
for i in range(len(bnum)):
    b = []
    for j in range(bnum[i]):
        b.append(evec[order[j]])

    b = np.array(b).T
    out.append(project(first5, b))
out = np.array(out)

plt.clf()
fig = plt.figure(figsize=(15, 3))
fig.patch.set_facecolor('white')
plt.subplot(151)
draw(out[0]+mean, 'original')
for i in range(len(bnum)):
    plt.subplot(152+i)
    draw(out[i+1]+mean, str(bnum[i])+'d')

plt.savefig('./fig/Q3.jpg')


#Q4
vec136 = []
tar136 = []
for i in range(10000):
    if target[i] == '1' or target[i] == '3' or target[i] == '6':
        vec136.append(data[i])
        tar136.append(target[i])

vec136 = np.array(vec136)
mean, cvec, evalue, evec, order = cpca(vec136)
b = []
for i in range(2):
    b.append(evec[order[i]]/(np.inner(evec[order[i]], evec[order[i]])**0.5))


b = np.array(b)

c = np.matmul(vec136, b.T)

color = []
plt.clf()

for i in range(len(tar136)):
    if tar136[i] == '1':
        color.append('blue')
    elif tar136[i] == '3':
        color.append('green')
    elif tar136[i] == '6':
        color.append('red')

fig = plt.figure(figsize=(6.4, 4.8))
fig.patch.set_facecolor('white')
plt.scatter(c[:, 0], c[:, 1], color=color)
plt.savefig('./fig/Q4.jpg')


#Q5

training = data[:10000, :]
a = data[10000]
index = omp(training, a, 5)


temp = []
plt.clf()
fig = plt.figure(figsize=(15, 3))
fig.patch.set_facecolor('white')
for i in range(5):
    plt.subplot(151+i)
    draw(data[index[i]], 'base'+str(i))
    temp.append(data[index[i]])

plt.savefig('./fig/Q5.jpg')


#Q6
a = data[10001]
training = data[:10000, :]

plt.clf()
fig = plt.figure(figsize=(15, 3))
fig.patch.set_facecolor('white')
plt.subplot(151)
draw(a, 'original')

basenum = [5, 10, 40, 200]

for i in range(len(basenum)):
    index = omp(training, a, basenum[i])
    temp = []
    for j in range(len(index)):
        temp.append(data[index[j]])

    temp = np.array(temp)

    plt.subplot(152+i)
    draw(project(a, temp.T), 's='+str(
        basenum[i])+' loss:'+str(int(np.linalg.norm(a-project(a, temp.T)))))

plt.savefig('./fig/Q6.jpg')



from sklearn import linear_model
clf = linear_model.Lasso(alpha=2.3, normalize=True, copy_X=True)

out=[]

vec8=[]
for i in range(length):
    if target[i]=='8':
        vec8.append(data[i])
        
vec8=np.array(vec8)
mean, cvec, evalue, evec, order = cpca(vec8)

last8=vec8[len(vec8)-1]

space=[]
for i in range(5):
    space.append(evec[order[i]])

space=np.array(space)
out.append(project(last8,space.T)+mean)



vec8+=mean
last8 = vec8[len(vec8)-1]

vec8 = vec8[:len(vec8)-1]

idx=omp(vec8,last8,5)
space = []
for i in range(5):
    space.append(vec8[idx[i]])

space = np.array(space)
out.append(project(last8, space.T))


clf.fit(vec8.T,last8)

idx=np.argsort(-abs(clf.coef_))


space = []
for i in range(5):
    space.append(vec8[idx[i]])

space = np.array(space)
out.append(project(last8, space.T))




out=np.array(out)

plt.clf()
fig = plt.figure(figsize=(9, 3))
fig.patch.set_facecolor('white')
for i in range(len(out)):
    plt.subplot(131+i)
    draw(out[i], str(i+1))

plt.savefig('./fig/Q7.jpg')





def soft(a,c,l):
    if c < -l:
        return (c+l)/a
    elif c>l:
        return (c-l)/a
    else:
        
        return 0




def lasso(orix,oriy,l):
    x=orix*1
    y=oriy*1
    ii,jj=x.shape

    for j in range(jj):
        x[:, j] /= (np.inner(x[:, j], x[:, j])**0.5)

    y/=(np.inner(y,y)**0.5)

    
    tempw=np.zeros(jj)
    a = np.zeros(jj)
    c = np.zeros(jj)
   
    s=x.T@x
    invs=np.linalg.pinv(s)
    w=invs@x.T@y
    
    
    
    while np.all(tempw!=w):
        tempw=w*1
        for j in range(jj):
            
            a[j] = 2*np.inner(x[:,j], x[:,j])
            c[j] = (2)*np.sum(x[:, j]@(y-(x@w)+(x[:, j]*w[j])))
            
            w[j]=soft(a[j],c[j],l)
    
    return w




z = lasso(vec8.T, last8, 1.5)
idx = np.argsort(-abs(z))


space = []
for i in range(5):
    space.append(vec8[idx[i]])

space = np.array(space)
draw(project(last8, space.T),'handcraft')

plt.savefig('./fig/Bonus.jpg')
