from alvin.util.vec import Vec
from alvin.util.matutil import coldict2mat
from alvin.util.matutil import mat2coldict
from alvin.util.solve import solve
from alvin.util.solve import solve2
from alvin.util.vecutil import list2vec


def is_superfluous(L, i):
    zero_like = 1e-14

    A = coldict2mat(L[:i] + L[i + 1:])
    b = L[i]
    u = solve(A, b)

    residual = b - A * u
    if (residual * residual < zero_like):
        return True
    else:
        return False


def is_independent(L):
    for i in range(len(L)):
        if is_superfluous(L, i):
            return False

    return True


def subset_basis(T):
    if is_independent(T):
        return T
    else:
        T.pop()
        return subset_basis(T)


def superset_basis(T, V):
    # step1: Initialize B to be equal to T.
    B = T

    # step2: add the vec from V by grow style
    for i in range(len(V)):
        v = V[i]
        if is_independent(B + [v]):
            B.append(v)
    return B


def exchange(S, A, z):
    result = []
    n_S = S[:]
    n_S.append(z)

    for i in range(len(n_S)):
        if is_superfluous(n_S, i) and n_S[i] not in A and n_S[i] != z:
            result.append(n_S[i])

    return result


def morph(S, B):
    result = []

    for i in range(len(B)):
        z = B[i]
        w = exchange(S, B, z)[0]
        result.append( (z, w) )
        S.append(z);
        S.remove(w);

    return result


def my_rank(L):
    return len(subset_basis(L))


def my_is_independent(L):
    return len(L) == my_rank(L)


def direct_sum_decompose2(U_basis, V_basis, w):
    S = [u+v for u in U_basis for v in V_basis]
    A = coldict2mat(S)
    u = solve(A, w)
    print(len(subset_basis(S)))
    return S


## Problem 9
def direct_sum_decompose(U_basis, V_basis, w):
    '''
    input:  A list of Vecs, U_basis, containing a basis for a vector space, U.
    A list of Vecs, V_basis, containing a basis for a vector space, V.
    A Vec, w, that belongs to the direct sum of these spaces.
    output: A pair, (u, v), such that u+v=w and u is an element of U and
    v is an element of V.

    >>> U_basis = [Vec({0, 1, 2, 3, 4, 5},{0: 2, 1: 1, 2: 0, 3: 0, 4: 6, 5: 0}), Vec({0, 1, 2, 3, 4, 5},{0: 11, 1: 5, 2: 0, 3: 0, 4: 1, 5: 0}), Vec({0, 1, 2, 3, 4, 5},{0: 3, 1: 1.5, 2: 0, 3: 0, 4: 7.5, 5: 0})]
    >>> V_basis = [Vec({0, 1, 2, 3, 4, 5},{0: 0, 1: 0, 2: 7, 3: 0, 4: 0, 5: 1}), Vec({0, 1, 2, 3, 4, 5},{0: 0, 1: 0, 2: 15, 3: 0, 4: 0, 5: 2})]
    >>> w = Vec({0, 1, 2, 3, 4, 5},{0: 2, 1: 5, 2: 0, 3: 0, 4: 1, 5: 0})
    >>> direct_sum_decompose(U_basis, V_basis, w) == (Vec({0, 1, 2, 3, 4, 5},{0: 2.0, 1: 4.999999999999972, 2: 0.0, 3: 0.0, 4: 1.0, 5: 0.0}), Vec({0, 1, 2, 3, 4, 5},{0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}))
    True
    '''
    joined_list = U_basis + V_basis
    u_vec = Vec(U_basis[0].D, {})
    v_vec = Vec(V_basis[0].D, {})
    rep = solve(coldict2mat(joined_list), w)
    for key in rep.f.keys():
        if (joined_list[key] in U_basis):
            u_vec = u_vec + rep.f[key] * joined_list[key]
        elif (joined_list[key] in V_basis):
            v_vec = v_vec + rep.f[key] * joined_list[key]

    return (u_vec, v_vec)


def is_invertible(M):
    if len(M.D[0]) != len(M.D[1]):
        return False

    T = mat2coldict(M)
    L = [T[v] for v in T]
    return is_independent(L)


def find_matrix_inverse(M):
    if is_invertible(M) == False:
        return None

    I = []
    for i in range(len(M.D[0])):
        row_list = []
        for j in range(len(M.D[1])):
            if i == j:
                row_list.append(1)
            else:
                row_list.append(0)
        I.append(list2vec(row_list))

    b = solve2(M, coldict2mat(I))
    return b


def find_triangular_matrix_inverse(M):
    if is_invertible(M) == False:
        return None

    I = []
    for i in range(len(M.D[0])):
        row_list = []
        for j in range(len(M.D[1])):
            if i == j:
                row_list.append(M[j,i])
            else:
                row_list.append(-M[j,i])
        I.append(list2vec(row_list))

    return coldict2mat(I)


M0 = [list2vec(v) for v in [[1,0.5,0.2,4], [0,1,0.3,0.9], [0,0,1,0.1], [0,0,0,1]]]
MM = coldict2mat(M0)
RM = find_triangular_matrix_inverse(MM)
print(RM)
RM2 = find_matrix_inverse(MM)


L0 = [list2vec(v) for v in [[1,3], [2,1], [3,1]]]
# print(is_invertible(coldict2mat(L0)))
L1 = [list2vec(v) for v in [[1,0,0,0], [0,2,0,0], [1,1,3,0], [0,0,1,4]]]
# print(is_invertible(coldict2mat(L1)))
L2 = [list2vec(v) for v in [[1,0,2], [0,1,1]]]
# print(is_invertible(coldict2mat(L2)))
L3 = [list2vec(v) for v in [[1,0], [0,1]]]
# print(is_invertible(coldict2mat(L3)))
L4 = [list2vec(v) for v in [[1,0,1], [0,1,1], [1,1,0]]]
# print(is_invertible(coldict2mat(L4)))

#print(coldict2mat(L4) * find_matrix_inverse(coldict2mat(L4)))


U_basis = [list2vec(v) for v in [[2,1,0,0,6,0], [11,5,0,0,1,0], [3,1.5,0,0,7.5,0]]]
Y_basis = [list2vec(v) for v in [[0,0,7,0,0,1], [0,0,15,0,0,2]]]
w = list2vec([2,5,0,0,1,0])
#print(direct_sum_decompose(U_basis,Y_basis,w))

S1=[list2vec(v) for v in [[2,4,0], [8,16,4], [0,0,7]]]
S2=[list2vec(v) for v in [[2,4,0], [8,16,4]]]
#print(my_is_independent(S1))
#print(my_is_independent(S2))
#print()

S1=[list2vec(v) for v in [[1,2,3], [4,5,6], [1.1,1.1,1.1]]]
S2=[list2vec(v) for v in [[1,3,0,0], [2,0,5,1], [0,0,1,0], [0,0,7,-1]]]
#print(my_rank(S1))
#print(my_rank(S2))
#print()


S=[list2vec(v) for v in [[0,0,5,3], [2,0,1,3], [0,0,1,0], [1,2,3,4]]]
A=[list2vec(v) for v in [[0,0,5,3], [2,0,1,3]]]
z=list2vec([0,2,1,1])
#print(exchange(S, A, z))

S=[ list2vec(v) for v in [[2,4,0],[1,0,3],[0,4,4],[1,1,1]] ]
B=[ list2vec(v) for v in [[1,0,0],[0,1,0],[0,0,1]] ]
# for (z,w) in morph(S,B):
#     print("injecting ", z)
#     print("ejecting ", w)
#     print()


a0 = Vec({'a', 'b', 'c', 'd'}, {'a': 1})
a1 = Vec({'a', 'b', 'c', 'd'}, {'b': 1})
a2 = Vec({'a', 'b', 'c', 'd'}, {'c': 1})
a3 = Vec({'a', 'b', 'c', 'd'}, {'a': 1, 'c': 3})
#print(is_superfluous([a0, a1, a2, a3], 0))
#print(is_superfluous([a0, a1, a2, a3], 1))
#print(is_superfluous([a0, a1, a2, a3], 2))
#print(is_superfluous([a0, a1, a2, a3], 3))
#print("--------")
#print(is_independent([a0, a3, a1, a2]))
#print(is_independent([a0, a1, a2]))
#print(is_independent([a0, a2, a3]))
#print(is_independent([a0, a1, a3]))
#print(is_independent([a0, a1, a2, a3]))

#print(subset_basis([ a0, a1, a2, a3 ]))
#print(subset_basis([ a0, a3, a1, a2 ]))

# print(superset_basis([ a0, a3 ], [ a0, a1, a2 ]))

b0 = Vec({'a', 'b', 'c'}, {'a': 0, 'b': 5, 'c': 3})
b1 = Vec({'a', 'b', 'c'}, {'a': 0, 'b': 2, 'c': 2})
b2 = Vec({'a', 'b', 'c'}, {'a': 1, 'b': 5, 'c': 7})

c0 = Vec({'a', 'b', 'c'}, {'a': 1, 'b': 1, 'c': 1})
c1 = Vec({'a', 'b', 'c'}, {'a': 0, 'b': 1, 'c': 1})
c2 = Vec({'a', 'b', 'c'}, {'a': 0, 'b': 0, 'c': 1})

#print(superset_basis([ b0, b1 ], [ c0, c1, c2 ]))
#
# print(superset_basis([ b0, b1, b2 ], [ c0, c1, c2 ]))
# print(superset_basis([ b0, b1 ], [ c0, c1, c2 ]))