from util.vec import Vec
from util.mat import Mat
from util.matutil import rowdict2mat

M = Mat(
    (
        {'a','b'}, {'@','#','?'}
    ),
    {
        ('a','@'):1, ('a','#'):2, ('a','?'):3,
        ('b','@'):10, ('b','#'):20, ('b','?'):30
    }
)

# Quiz 5.3.1
def mat2vec(M):
    return Vec({(r,s) for r in M.D[0] for s in M.D[1]}, M.f)
print("\nQuiz 5.3.1")
mv = mat2vec(M)
print(mv.f)

# Quiz 5.4.2
def transpose(M):
    return Mat((M.D[1], M.D[0]), {(q,p):v for (p,q), v in M.f.items()})
print("\nQuiz 5.4.2")
M2 = transpose(M)
print(M2)

# Example 5.5.10
D = { 'metal', 'concrete', 'plastic', 'water', 'electricity' }
v_gnome = Vec(D, {'concrete':1.3, 'plastic':0.2, 'water':0.8, 'electricity':0.4})
v_hoop = Vec(D, {'plastic':1.5, 'water':0.4, 'electricity':0.3})
v_slinky = Vec(D, {'metal':0.25, 'water':0.2, 'electricity':0.3})
v_putty = Vec(D, {'plastic':0.3, 'water':0.7, 'electricity':0.5})
v_shooter = Vec(D, {'metal':0.15, 'plastic':0.5, 'water':0.4, 'electricity':0.8})

rowdict = {'gnome':v_gnome, 'hoop':v_hoop, 'slinky':v_slinky, 'putty':v_putty, 'shooter':v_shooter}
M = rowdict2mat(rowdict)
print(M)

R = {'gnome', 'hoop', 'slinky', 'putty', 'shooter'}
u = Vec(R, {'putty':133, 'gnome':240, 'slinky':150, 'hoop':55, 'shooter':90})
print(u*M)

# Example 5.5.15
# C = {'metal','concrete','plastic','water','electricity'}
# b = Vec(C, {'water':373.1, 'concrete':312.0, 'plastic':215.4, 'metal':51.0, 'electricity':356.0})
# solution = util.solver(M.transpose(), b)
# print(solution)
print(M.transpose())