from alvin.util.image_mat_util import file2image
from alvin.util.image_mat_util import image2mat
# from alvin.util.image_mat_util import svglist2display
from alvin.util.image_mat_util import mat2display

img1 = file2image("res/a.png")
img2 = file2image("res/b.png")
mat1 = image2mat(img1)
mat2 = image2mat(img2)

# mat[0] 위치, mat[1] RGB
#print(mat1[0])
#print(mat1[1])

# 결과
res1 = mat1[1] * 0.5
res2 = mat2[1] * 0.5
res = res1 + res2

mat2display(mat1[0], res)
# mat2svg(mat1[0], res)

# list = []
# size = 9
#
# for i in range(size + 1):
#     dist = i / size
#
#     res1 = mat1[1] * (1 - dist)
#     res2 = mat2[1] * dist
#     res = res1 + res2
#
#     list.append([ mat1[0], res ])
#
# print(len(list))
# svglist2display(list, "res/effect.js")