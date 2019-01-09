import cv2
import numpy as np
import time
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
import math


from lib import *




def mask_noise(im, ori, ver, th):

    h = im.shape[0]
    w = im.shape[1]

    dx = w/ori
    dy = h/ver
    

    mask = np.zeros((ver+1,ori+1))

    for y in range(0,h):
        for x in range(0,w):
            if im[y,x]>0:
                a = int(math.ceil(float(x)/float(dx)))-1
                b = int(math.ceil(float(y)/float(dy)))-1

                mask[b,a] += 1

    for y in range(0,ver):
        for x in range(0,ori):
            if mask[y,x]>th:
                mask[y,x] = 1
            else:
                mask[y,x] = 0
    
    return mask

def apply_mask(im,mask):

    h = im.shape[0]
    w = im.shape[1]

    try:
        for y in range(0,h):
            for x in range(0,w):
                im[y,x] *= mask[y,x]
    except:
        print("problema dimensioni")
    return im

def print_mask(im,mask):

    h = im.shape[0]
    w = im.shape[1]

    try:
        for y in range(0,h):
            for x in range(0,w):
                if im[y,x]==0:
                    im[y,x] += 100*mask[y,x]
    except:
        print("problema dimensioni")
    return im

def matrix_expand(matrix, sh, sw):

    mh = matrix.shape[0]
    mw = matrix.shape[1]

    rh = sh/mh
    rw = sw/mw

    print(rh,rw)

    res = []

    for ele in matrix:
        t = np.repeat(ele,rw)
        for c in range(0,rh):
            res.append(t)

    return np.array(res)
    
def create_network(db):

    imgs = [ele["img"] for ele in db]
    values = [ele["value"] for ele in db]

    train_x, test_x, train_y, test_y = train_test_split(imgs,values)


    res = []

    for k in range(1, 30, 1):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(train_x, train_y)

        score = model.score(test_x, test_y)
        res.append({"score":score,"k":k})
        print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    
    k = sorted(res, key=lambda key: key['score'])[::-1][0]["k"]
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(train_x, train_y)

    return model




fr = open("dba.pkl", 'rb')
db = pickle.load(fr)
fr.close()

fr = open("dbb.pkl", 'rb')
db += pickle.load(fr)
fr.close()

fr = open("dbc.pkl", 'rb')
db += pickle.load(fr)
fr.close()





for ele in db:
    ele["img"] = make_16x16(ele["img"]).ravel()


net = create_network(db)


im = cv2.imread("/home/java/Scrivania/c.jpg")
im = cv2.GaussianBlur(im,(5,5),0)
im = resize_h(im,1024)
im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
im = zone_clean(im,3,3,4)




boxes = img_as_boxes(im,30,100000)

for box in boxes:
    correct_box(im,box,3,3)

boxes = img_as_boxes(im,30,100000)

'''
mask = mask_noise(im, 128, 128,30)
print("maschera")
print(mask)

mask = matrix_expand(mask,1024,975)
print(mask.shape)


print(mask)

mask = np.ones((1024,1024))
im = apply_mask(im,mask)
im = print_mask(im,mask)
'''


#im = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)


cv2.imwrite("/home/java/Scrivania/res.jpg",im)


A = as_generic_matrix(im,boxes,net,3,3)


print(A)
print("DET = "+str(int(np.linalg.det(A))))
