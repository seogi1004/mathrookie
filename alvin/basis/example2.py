from alvin.util.vec import Vec
from alvin.util.matutil import coldict2mat
from alvin.util.matutil import rowdict2mat
from alvin.util.matutil import mat2coldict
from alvin.util.solve import solve
from alvin.util.solve import solve2
from alvin.util.vecutil import list2vec

def echelon_form(rowlist):
    rowlist = rowlist[:]
    col_label_list = sorted(rowlist[0].D, key=hash)
    new_rowlist = []
    rows_left = set(range(len(rowlist)))

    row_label_list = col_label_list
    row_labels = set(range(len(rowlist)))
    M_rowlist = [Vec(row_labels, {row_label_list[i]:1}) for i in range(len(rowlist))]

    new_M_rowlist = []
    for c in col_label_list:
        rows_with_nonzero = [r for r in rows_left if rowlist[r][c] != 0]
        if rows_with_nonzero != []:
            pivot = rows_with_nonzero[0]
            rows_left.remove(pivot)
            new_M_rowlist.append(M_rowlist[pivot])
            new_rowlist.append(rowlist[pivot])

            for r in rows_with_nonzero[1:]:
                multiplier = rowlist[r][c]/rowlist[pivot][c]
                rowlist[r] -= multiplier*rowlist[pivot]
                M_rowlist[r] -= multiplier*M_rowlist[pivot]

    for r in rows_left:
        new_M_rowlist.append(M_rowlist[r]);

    return new_M_rowlist, new_rowlist

A = [list2vec(v) for v in [[0,2,3,4,5], [0,0,0,3,2], [1,2,3,4,5], [0,0,0,6,7], [0,0,0,9,8]]]
M,MA = echelon_form(A)
print(rowdict2mat(M))
print(rowdict2mat(MA))
print(rowdict2mat(M) * rowdict2mat(A))

#print(rowdict2mat(M) * rowdict2mat(A))

A = [list2vec(v) for v in [[1,2,3,4,5], [0,2,3,4,5], [0,0,0,3,2], [0,0,0,6,7], [0,0,0,9,8]]]
M = [list2vec(v) for v in [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,-2,1,0], [0,0,0,0,1]]]
RM = [list2vec(v) for v in [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,2,1,0], [0,0,0,0,1]]]
# print(rowdict2mat(M) * rowdict2mat(A))
# print(rowdict2mat(RM) * rowdict2mat(M) * rowdict2mat(A))


A = [
    list2vec(v) for v in [
        [1, 2, 3, 4, 5],
        [0,2,3,4,5],
        [0,0,0,3,2],
        [0,0,0,6,7],
        [0,0,0,9,8]
    ]
]
M = [
    list2vec(v) for v in [
        [1,0,0,0,0],
        [0,1,0,0,0],
        [0,0,1,0,0],
        [0,0,-2,1,0],
        [0,0,-3,0,1]
    ]
]
M0 = [
    list2vec(v) for v in [
        [1,0,0,0,0],
        [0,1,0,0,0],
        [0,0,1,0,0],
        [0,0,0,1,0],
        [0,0,0,-2/3,1]
    ]
]

#print(rowdict2mat(M) * rowdict2mat(A))
#print(rowdict2mat(M0) * rowdict2mat(M) * rowdict2mat(A))