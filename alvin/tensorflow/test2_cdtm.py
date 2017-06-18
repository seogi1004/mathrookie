from alvin.util.vecutil import list2vec
from alvin.util.matutil import coldict2mat
from alvin.util.matutil import mat2coldict
import random as rd

def cost(X, Y, W):
    # simplified hypothesis (b를 제거함
    # hypothesis = X * W (Wx)
    # hypothesis는 계산의 편의성을 위해 3x1 행렬로 만든다.
    H = X * W
    dim = len(H.D[0])

    # cost = tf.reduce_mean(tf.square(hypothesis - Y))
    H -= Y
    cost = 0
    for i in range(dim):
        cost += (H[i, 0] ** 2) / dim

    W[0, 0] = cost
    return W

def cost2(X, Y, W):
    # simplified hypothesis (b를 제거함
    # hypothesis = X * W (Wx)
    # hypothesis는 계산의 편의성을 위해 3x1 행렬로 만든다.
    dim = len(X.D[0])
    H = (X * W) - Y

    # cost = tf.reduce_mean(tf.square(hypothesis - Y))
    hypothesis_vec = mat2coldict(H)[0]
    W[0, 0] = (hypothesis_vec * hypothesis_vec) / dim

    return W

def minimize(X, Y, W, sigma=0.01):
    dim = len(X.D[0])
    H = X * W - Y

    for i in range(dim):
        H[i, 0] = H[i, 0] * X[i, 0]

    gradient_vec = mat2coldict(H)[0]
    gradient = (gradient_vec * gradient_vec) / dim

    W[0,0] = W[0,0] - sigma * gradient

    return W

X = coldict2mat([list2vec(v) for v in [ [1, 2, 3] ]])
Y = coldict2mat([list2vec(v) for v in [ [1, 2, 3] ]])
W = coldict2mat([list2vec(v) for v in [ [rd.random()] ]])

tw = W
for i in range(10):
    tw = minimize(X, Y, tw)
    print(tw[0,0])