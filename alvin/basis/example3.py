
import random
from itertools import permutations

from alvin.util.vecutil import list2vec
from alvin.util.GF2 import one
from alvin.basis.example import is_independent

from alvin.util.bitutil import str2bits
from alvin.util.bitutil import bits2str
from alvin.util.bitutil import bits2mat
from alvin.util.bitutil import mat2bits


def is_prime(N):
    return True

def factor(N):
    return 1,2

def prime_factorize(N):
    if is_prime(N):
        return [N]

    a,b = factor(N)
    return prime_factorize(a) + prime_factorize(b)

## 1: (Task 1) Choosing a Secret Vector
def randGF2(): return random.randint(0,1)*one

a0 = list2vec([one, one,   0, one,   0, one])
b0 = list2vec([one, one,   0,   0,   0, one])

def choose_secret_vector(s,t):
    u = list2vec([randGF2() for x in range(6)])
    if a0 * u == s and b0 * u == t:
        return u
    else:
        return choose_secret_vector(s, t)

s = randGF2()
t = randGF2()
print(s)
print(t)
print(choose_secret_vector(s, t))

strmat = bits2mat(str2bits("Rosebud"), 2)
ulist = []
for j in strmat.D[1]:
    s = strmat[0,j]
    t = strmat[1,j]
    ulist.append(choose_secret_vector(s, t))

print(ulist)



## Problem 2
# Give each vector as a Vec instance
secret_a0 = list2vec([one, one, 0, one, 0, one])
secret_b0 = list2vec([one, one, 0, 0, 0, one])
secret_a1 = list2vec([0, 0, 0, 0, one, 0])
secret_b1 = list2vec([one, one, 0, one, 0, 0])
secret_a2 = list2vec([0, one, 0, one, 0, one])
secret_b2 = list2vec([0, 0, one, 0, one, 0])
secret_a3 = list2vec([one, 0, 0, 0, one, 0])
secret_b3 = list2vec([0, one, one, 0, one, one])
secret_a4 = list2vec([one, one, one, one, 0, 0])
secret_b4 = list2vec([one, 0, one, one, 0, one])

# solve가 GF를 풀지못함
def choose_rand_vector(randNum=4, vecLength=6):
    r = []
    r.append(secret_a0)
    r.append(secret_b0)
    while randNum > 0:
        u = list2vec([randGF2() for i in range(vecLength)])
        r.append(u)
        if is_independent(r):
            randNum -= 1
        else:
            r.pop()
    return [(r[0], r[1]), (r[2], r[3]), (r[4], r[5])]


def choose_pair():
    r = []
    while True:
        r1 = choose_rand_vector(randNum=4, vecLength=6)
        r2 = choose_rand_vector(randNum=4, vecLength=6)
        r3 = r1 + r2
        r3.remove(r1[0])
        r3.remove(r2[0])
        r4 = list(permutations(r3, 2))
        r5 = [list(r1[0]) + list(x[0]) + list(x[1]) for x in r4]
        for i in r5:
            if not is_independent(i): break
        return r3


'''
k=choose_pair()
z=[]
k=[list(x) for x in k]
#print(k)
[print(x[0].f.values(),x[1].f.values()) for x in k]
#[print(list(a.f.values())) for a in r]
'''