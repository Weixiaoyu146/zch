import os

Nums = 2000
questions_pelvis = [109, 120, 166, 167, 168, 170, 264, 274, 361, 396, 411, 453,
             465, 476, 496, 559, 594, 633, 636, 656, 689]
valueless_pelvis = [44, 92, 97, 102, 108, 125, 129, 133, 146, 224, 228, 246, 250,
             280, 310, 381, 401, 420, 433, 436, 456, 471, 514, 535, 540,
             543, 584, 597, 598, 605, 609, 624, 639, 679, 683, 697, 707]

# 获取存在的图像索引
def get_indexs(path):
    exist = []
    g = os.walk(path)

    for p, dirs, files in g:
        for i in range(0, Nums):
            pathA = str(i) + 'a.jpg'
            if pathA in files:
                exist.append(i)

    # valueless = valueless_pelvis + questions_pelvis
    # exist = [i for i in exist if i not in valueless]

    return exist

# 筛除不存在的
def find_unexist_indexs(path):
    g = os.walk(path)
    fo = open("unexistence.txt", "w")

    for p, dirs, files in g:
        for i in range(1, Nums):
            pathA = str(i) + 'a.jpg'
            pathP = str(i) + 'p.jpg'
            if pathA not in files:
                fo.write(pathA.split('.')[0] + '\n')
            if pathP not in files:
                fo.write(pathP.split('.')[0] + '\n')
    fo.close()

    return
