import numpy as np
from pandas import DataFrame as df


X = [True,True,True,True,True,False,False,False,False,True]
X = np.array(X)
print(X)
X[X==True] = 1
X[X==False] = 0
print(X)

A=[0.15765254 ,0.00161707 ,0.00800906 ,0.02944399 ,0.00065909 ,0.00915422,-0.00157297, 0.03086107,0.10459237 ,0.01617813 ,0.39146412,0.01]
B=[0.15765254 ,0.00161707 ,0.00800906 ,0.02944399 ,0.00065909 ,0.00915422,-0.00157297, 0.03086107,0.10459237 ,0.01617813 ,0.39146412,0.01]
A = np.array(A)
print(A)
A[A<0.01] = 0
A[A>=0.01] = 1
print(A)
C=[]
#A=int(A)
i=0
for i in range(0, len(A)):
    if(A[i]==0):
        C.append(i)
print(C)
B = B.drop(C,axis=1)
print(B)
