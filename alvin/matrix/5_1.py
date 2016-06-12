from util.vec import Vec
from util.mat import Mat
from util.matutil import listlist2mat

# Quiz 5.1.1
r1 = [ [ 0 for j in range(4) ] for i in range(3) ]
print("\nQuiz 5.1.1")
print(r1)

# Quiz 5.1.2
r2 = [ [ i-j for i in range(3) ] for j in range(4) ]
print("\nQuiz 5.1.2")
print(r2)

# Quiz 5.1.3
r3 = Vec({'a','b'}, {'a':3,'b':30})
print("\nQuiz 5.1.3")
print(r3)

# Quiz 5.1.5
r4 = {
    '#': Vec({'a', 'b'}, {'a':2, 'b':20}),
    '@': Vec({'a', 'b'}, {'a':1, 'b':10}),
    '?': Vec({'a', 'b'}, {'a':3, 'b':30})
}
print("\nQuiz 5.1.5")
print(r4)

# 5.1.4
M = Mat(
    (
        {'a','b'}, {'@','#','?'}
    ),
    {
        ('a','@'):1, ('a','#'):2, ('a','?'):3,
        ('b','@'):10, ('b','#'):20, ('b','?'):30
    }
)
print("\n5.1.4")
print(M)

# Quiz 5.1.7
M2 = Mat(
    (
        {'a','b','c'}, {'a','b','c'}
    ),
    {
        ('a','a'):1, ('b','b'):1, ('c','c'):1
    }
)
print("\nQuiz 5.1.7")
print(M2)

# Quiz 5.1.8
def identity(D):
    return Mat((D,D), {(d,d):1 for d in D})
U1 = identity({'a','b','c'})
print("\nQuiz 5.1.8")
print(U1)

# Quiz 5.1.9
def mat2rowdict(A):
    return { r:Vec(A.D[1], {c:A[r,c] for c in A.D[1]}) for r in A.D[0] }
R1 = mat2rowdict(M)
print("\nQuiz 5.1.9")
print(R1)

# Quiz 5.1.10
def mat2coldict(A):
    return { c:Vec(A.D[0], {c:A[r,c] for r in A.D[0]}) for c in A.D[1] }
R2 = mat2coldict(M)
print("\nQuiz 5.1.10")
print(R2)

# 5.1.7
A = listlist2mat([[10,20,30,40],[50,60,70,80]])
print("\n5.1.7")
print(A)



