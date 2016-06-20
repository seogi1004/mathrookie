from alvin.util.GF2 import one
from alvin.util.matutil import listlist2mat
from alvin.util.matutil import mat2coldict
from alvin.util.vecutil import list2vec
from alvin.util.bitutil import str2bits
from alvin.util.bitutil import bits2str
from alvin.util.bitutil import bits2mat
from alvin.util.bitutil import mat2bits
from alvin.util.bitutil import noise

# Task 5.14.1
G = listlist2mat([
    [ one, 0, one, one ],
    [ one, one, 0, one ],
    [ 0, 0, 0, one ],
    [ one, one, one, 0 ],
    [ 0, 0, one, 0 ],
    [ 0, one, 0, 0 ],
    [ one, 0, 0, 0 ]
])

p = list2vec([
    one, 0, 0, one
])

# 메시지 [1,0,0,1]을 부호화한 값
c = G * p

print("Task 5.14.1")
print(G)
print(c)
print("\n")


# Task 5.14.2
R = listlist2mat([
    [ 0, 0, 0, 0, 0, 0, one ],
    [ 0, 0, 0, 0, 0, one, 0 ],
    [ 0, 0, 0, 0, one, 0, 0 ],
    [ 0, 0, one, 0, 0, 0, 0 ]
])

print("Task 5.14.2")
print(R)
print(R * c)
print(R * G)
print("\n")


# Task 5.14.3
H = listlist2mat([
    [ 0, 0, 0, one, one, one, one ],
    [ 0, one, one, 0, 0, one, one ],
    [ one, 0, one, 0, one, 0, one ]
])

print("Task 5.14.3")
print(H * G)
print("\n")


# Task 5.14.4
def find_error(c):
    return H * c

print("Task 5.14.4")
e1 = find_error(list2vec([ one, 0, one, one, 0, one, one ]))
print(e1)
e2 = find_error(list2vec([ one, 0, one, one, 0, one, 0 ]))
#e2 = find_error(listlist2mat([
#    [one],
#    [0],
#    [one],
#    [one],
#    [0],
#    [one],
#    [0]
#]))
print(e2)
print("\n")


# Task 5.14.5
print("Task 5.14.5")
def find_error_matrix(e):
    index = 0
    for i in range(3):
        if(e[i] == one):
            index += 2 ** (2 - i)
    return index

print(find_error_matrix(e1))
print(find_error_matrix(e2))
print("\n")


# Task 5.14.6
print("Task 5.14.6")
s1 = "I'm trying to free your mind, Neo. But I can only show you the door. You're the one that has to walk through it."
s2 = str2bits(s1)
print(s1)
print(s2)
print(bits2mat(s2))
print(bits2str(s2))
print("\n")


# Task 5.14.7
print("Task 5.14.7")
P = bits2mat(s2)
print(P)
print("\n")


# Task 5.14.9
print("Task 5.14.9")
E = noise(P, 0.02)
print(bits2str(mat2bits(E + P)))
print("\n")


# Task 5.14.10
print("Task 5.14.10")
print(P)
C = G * P
print(bits2str(mat2bits(C)))
print("\n")


# Task 5.14.11
print("Task 5.14.11")
E = noise(C, 0.02)
CTILDE = E + C
print(bits2str(mat2bits(CTILDE)))
# 복호화 노이즈
print(bits2str(mat2bits(R * CTILDE)))
# 복호화 원문
print(bits2str(mat2bits(R * C)))
print("\n")


# Task 5.14.12
print("Task 5.14.12")
print(C)
print(CTILDE)
print("======================\n")
#print(R * CTILDE)
#print(bits2str(mat2bits(R * CTILDE)))
#print("----------------------\n")
print(find_error(C))
print(find_error(CTILDE))
#print("----------------------\n")
print("\n")


# Task 5.14.13
print("Task 5.14.13")
def correct(c):
    print("Before: " + bits2str(mat2bits(R * c)))

    errordict = mat2coldict(find_error(c))
    size = len(errordict)

    # 영행렬 생성
    E = c.copy()
    for j in range(size):
        for i in range(7):
            E[i,j] = 0

    # 에러 신드롬 생성
    for j in range(size):
        i = find_error_matrix(errordict[j])
        if i > 0:
            E[i - 1, j] = one

    result = c + E

    print("After: " + bits2str(mat2bits(R * result)))

correct(CTILDE)
print("\n")