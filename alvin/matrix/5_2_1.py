from util.vec import Vec
from util.mat import Mat
from util.matutil import rowdict2mat

# Example 5.5.10
D = { 'metal', 'concrete', 'plastic', 'water', 'electricity' }
v_gnome = Vec(D, {'concrete':1.3, 'plastic':0.2, 'water':0.8, 'electricity':0.4})

rowdict = {'gnome':v_gnome }
M = rowdict2mat(rowdict)
print(M)

R = {'gnome' }
u = Vec(R, { 'gnome':100 })
print(100 * v_gnome)

# Example 5.5.15
C = {'metal','concrete','plastic','water','electricity'}
b = Vec(C, {'water':373.1, 'concrete':312.0, 'plastic':215.4, 'metal':51.0, 'electricity':356.0})
print()