import numpy as np

def relu(x):
    return np.maximum(0, x)

# 3.3.1
A = np.array([1, 2, 3, 4])
print(A)
print(np.ndim(A))
print(A.shape)
print(A.shape[0])
print("\n")

B = np.array([[1,2], [3,4], [5,6]])
print(B)
print(np.ndim(B))
print(B.shape)
print("\n")

# 3.3.2
A = np.array([ [1,2], [3,4] ])
print(A.shape)

B = np.array([ [5,6], [7,8] ])
print(B.shape)

print(np.dot(A, B))
print(A * B)
print(A * 2)
print("\n")

A = np.array([ [1,2,3], [4,5,6] ])
print(A.shape)

B = np.array([ [1,2], [3,4], [5,6] ])
print(B.shape)
print(np.dot(A, B))
print("\n")

A = np.array([ [1,2], [3,4], [5,6] ])
B = np.array([7,8])
print(A * B)
print(A.dot(B))
print("\n")

X = np.array([1, 2])
W = np.array([ [1, 3, 5], [2, 4, 6] ])
print(np.dot(X, W))
