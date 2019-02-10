import cv2
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
import math
import sys

from lib import *



def get_matrix_shape(boxes):
    h = 1
    w = 1

    boxes = sorted(boxes, key=lambda k: k['y'])
    for c in range(0, len(boxes)-1):
        if abs(boxes[c]["y"]-boxes[c+1]["y"])>40:
            print("H")
            h+=1

    boxes = sorted(boxes, key=lambda k: k['x'])
    for c in range(0, len(boxes)-1):
        if abs(boxes[c]["x"]-boxes[c+1]["x"])>40:
            print("W")
            w+=1

    print(w,h)



    return {'h': h, 'w': w}

        
    




def box_mask(im, boxes):
    h = im.shape[0]
    w = im.shape[1]

    mask = np.zeros([h,w])

    for box in boxes:
        
        for a in range(box["x"],box["x"]+box["w"]):
            for b in range(box["y"],box["y"]+box["h"]):
                mask[b,a] = 255


    for a in range(0,w):
        for b in range(0,h):
            if mask[b,a]==0:
                im[b,a] = 0

    return im



    
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



'''
fr = open("dba.pkl", 'rb')
db = pickle.load(fr, encoding='latin1')
fr.close()

fr = open("dbb.pkl", 'rb')
db += pickle.load(fr, encoding='latin1')
fr.close()

fr = open("dbc.pkl", 'rb')
db += pickle.load(fr, encoding='latin1')
fr.close()





for ele in db:
    ele["img"] = make_16x16(ele["img"]).ravel()



net = create_network(db)
'''

#load from file predefined network
fr = open("network.pkl","rb")
net = pickle.load(fr)
fr.close()



'''
#saving network
fw = open("network.pkl","wb")
fw.write(pickle.dumps(net))
fw.close()
'''


inp = input("input image\n")

im = cv2.imread(inp)
im = cv2.GaussianBlur(im,(5,5),0)
im = resize_h(im,1024)
im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
im = zone_clean(im,3,3,4)




boxes = img_as_boxes(im,30,100000)
im = box_mask(im, boxes)




m = get_matrix_shape(boxes)
print("trovata matrice %dx%d" % (m["h"],m["w"]))

A = as_generic_matrix(im,boxes,net,m["w"],m["h"])


print(A)
if m["h"] == m["w"]:
    print("DET = "+str(int(np.linalg.det(A))))
