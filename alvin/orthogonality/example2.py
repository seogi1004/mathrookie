from alvin.util.vecutil import list2vec
from alvin.orthogonality.orthogonalization import orthogonalize
from alvin.orthogonality.orthogonalization import aug_orthogonalize
from alvin.orthogonality.orthogonalization import project_orthogonal

# vlist = [ list2vec(v) for v in [[2,0,0], [1,2,2], [1,0,2]] ]
# print(orthogonalize(vlist))
#
# vlist = [ list2vec(v) for v in [[8,-2,2], [4,2,4]] ]
# print(orthogonalize(vlist))
#
# vlist = [ list2vec(v) for v in [[1,0,2], [1,0,2], [2,0,0]] ]
# print(orthogonalize(vlist))

# 10.4 첫번째 방법
vlist = [ list2vec(v) for v in [[8,-2,2], [4,2,4]] ]
vstartlist = orthogonalize(vlist)
print(project_orthogonal(list2vec([ 5, -5, 2 ]), vstartlist))
print("\n")

# 10.4 두번째 방법
vlist = [ list2vec(v) for v in [[8,-2,2], [4,2,4], [5,-5,2]] ]
print(orthogonalize(vlist))
print("\n")

# 10.3.7의 u 첨가행렬 구하기 (10.5.3 augmented_orthogonalize)
vlist = [ list2vec(v) for v in [[2,0,0], [1,2,2], [1,0,2]] ]
print(aug_orthogonalize(vlist)[0])
print(aug_orthogonalize(vlist)[1])
print("\n")

# example 10.6.7
L = [ list2vec(v) for v in [[8,-2,2], [0,3,3], [1,0,0], [0,1,0], [0,0,1]] ]
Lstart = orthogonalize(L)