
# coding: utf-8

# In[1]:

import numpy as np


# In[2]:

N, D_in, H, D_out = 4, 2, 2, 1


# In[3]:

x = np.array([[0,0],[1,0],[0,1],[1,1]])
y = np.array([0,1,1,0]).reshape(4,1)
w1, w2 = np.random.rand(D_in, H), np.random.rand(H,D_out)
b1, b2 = np.random.rand(N,H), np.random.rand(N,D_out)


# In[5]:

for t in range(10000):

        #ForwardPropogation
        h=x.dot(w1) + b1
        a=np.maximum(h,0)
        y_pred = a.dot(w2) + b2
        loss = np.square(y_pred - y).sum()
        print(t,loss)


        #BackProp
        grad_y_pred = 2*(y_pred - y)
        grad_w2 = a.T.dot(grad_y_pred)
        grad_b2 = grad_y_pred
        grad_a = grad_y_pred.dot(w2.T)
        grad_h = grad_a.copy()
        grad_h[h<0] = 0
        grad_w1 = x.T.dot(grad_h)
        grad_b1 = grad_h

        w1 -= 0.001*grad_w1
        w2 -= 0.001*grad_w2
        b1 -= 0.001*grad_b1
        b2 -= 0.001*grad_b2


# In[12]:

x = np.array([[0,0],[1,0],[0,1],[1,1]])


# In[13]:

h=x.dot(w1) + b1
a=np.maximum(h,0)
y_pred = a.dot(w2) + b2


# In[14]:

print(y_pred)


# In[ ]:





