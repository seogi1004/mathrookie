
import random
from itertools import permutations

from alvin.util.vecutil import list2vec
from alvin.util.matutil import rowdict2mat
from alvin.util.matutil import coldict2mat
from alvin.util.GF2 import one
from alvin.basis.example import is_independent

from alvin.util.bitutil import str2bits
from alvin.util.bitutil import bits2str
from alvin.util.bitutil import bits2mat
from alvin.util.bitutil import mat2bits
from alvin.util.echelon import transformation_rows

from alvin.util.factoring_support import intsqrt
from alvin.util.factoring_support import gcd
from alvin.util.factoring_support import dumb_factor
from alvin.util.factoring_support import primes
from alvin.util.factoring_support import prod
from math import sqrt
from math import ceil
from math import pow
from random import randint
from alvin.util.vec import Vec

def is_prime(N):
    return True

def factor(N):
    return 1,2

def prime_factorize(N):
    if is_prime(N):
        return [N]

    a,b = factor(N)
    return prime_factorize(a) + prime_factorize(b)

def find_divisor(N):
    for i in range(2, N):
        if N % i == 0:
            return i

def root_method(N):
    res = 0
    a = ceil(sqrt(N))

    while(True):
        b = sqrt(a ** 2 - N)

        if(b % 1 == 0):
            res = a - b
            break

        a += 1

    return res

def int2GF2(i):
    if i % 2 == 0:
        return 0
    return one

def make_Vec(primeset, list):
    res = {}
    for k,v in list:
        res[k] = int2GF2(v)

    return Vec(primeset, res)

def find_candidates(N, primeset):
    roots = []
    row_list = []

    for x in range(intsqrt(N) + 2,N):
        a = x*x - N
        res = dumb_factor(a, primeset)

        if len(roots) <= len(primeset) and len(res) > 0:
            roots.append(x)
            row_list.append(make_Vec(primeset, res))

    return roots, row_list

def find_a_and_b(v, roots, N):
    primeset = { key for key in v.D if v[key] != 0 }

    alist = [ roots[index] for index in primeset ]
    print(alist)
    a = prod(alist)

    clist = [ x*x - N for x in alist ]
    c = prod(clist)

    return a, intsqrt(c)


print("\ntask 8.8.1")
print(root_method(55))
print(root_method(77))
print(root_method(146771))
# print(root_method(118))

# task 8.8.2
r = randint(0, 100)
s = randint(0, 100)
t = randint(0, 100)

a = r * s
b = s * t

d = gcd(a, b)
print("\ntask 8.8.2")
print(d)
print(a % d == 0)
print(b % d == 0)
print(d >= s)


# task 8.8.3
N = 367160330145890434494322103
a = 67469780066325164
b = 9429601150488992
print("\ntask 8.8.3")
print((a*a - b*b) % N)


# task 8.8.4
print("\ntask 8.8.4-1")
primeset = {2,3,5,7}
print(dumb_factor(75, primeset))
print(dumb_factor(30, primeset))
print(dumb_factor(1176, primeset))
print(dumb_factor(2*17, primeset))
print(dumb_factor(2*3*5*19, primeset))


# task 8.8.4
print("\ntask 8.8.4-2")
primeset = {2,3,5,7,11,13}
x1 = 12
x2 = 154
x3 = 2 * 3 * 3 * 3 * 11 * 11 * 13
x4 = 2 * 7
x5 = 2 * 3 * 5 * 7 * 19
print(dumb_factor(x1, primeset))
print(dumb_factor(x2, primeset))
print(dumb_factor(x3, primeset))
print(dumb_factor(x4, primeset))
print(dumb_factor(x5, primeset))


# task 8.8.5
print("\ntask 8.8.5")
print(int2GF2(3))
print(int2GF2(4))


# task 8.8.6
print("\ntask 8.8.6")
primeset = {2,3,5,7,11}
print(make_Vec(primeset, [(3,1)]))
print(make_Vec(primeset, [(2,17),(3,0),(5,1),(11,3)]))


# task 8.8.7
print("\ntask 8.8.7")
N = 2419
data = find_candidates(N, primes(32))
A = data[1]
M = transformation_rows(A)

print(rowdict2mat(A))
print(rowdict2mat(M))
print(rowdict2mat(M) * rowdict2mat(A))

# 53,77
print(A[2] + A[10])
print(gcd(53*77 - 2*9*5*13, N))

# 52,67,71
print(A[1] + A[7] + A[9])
print(gcd(52*67*71 - 2*9*5*19*23, N))

# print(new_row_list)
# print(new_row_list[0] + new_row_list[1])

# task 8.8.10
print("\ntask 8.8.10")
res1 = find_a_and_b(M[11], data[0], N)
print(res1)
# res2 = find_a_and_b(M[10], data[0], N)
# print(res2)
# print(53*77, 2*9*5*13)

# # task 8.8.11
# print("\ntask 8.8.11")
# # N = 2461799993978700679
# N = 2419
#
# print("\ntask 8.8.11 -- 1")
# # data = find_candidates(N, primes(10000))
# primeset = primes(32)
# data = find_candidates(N, primeset)
# A = data[1]
#
# print("\ntask 8.8.11 -- 2")
# M = transformation_rows(A, sorted(primeset, reverse=True))
#
# print("\ntask 8.8.11 -- 3")
# res = find_a_and_b(M[len(M) - 1], data[0], N)
# print(res)
# print("\ntask 8.8.11 -- end")
#
# a = 51
# b = 182
# print("----")
# print(a**2 - (a**2-N))
#
# a = 52
# b = 285
# print(a**2 - (a**2-N))