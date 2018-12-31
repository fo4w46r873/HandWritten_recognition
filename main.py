import cv2
import numpy as np
import time
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
import math


from lib import *





def clean_img_to_matrix(im,net):
    boxes = img_as_boxes(im,50,3000)

    for box in boxes:
        correct_box(im,box,3,3)

    #cv2.imwrite("/home/java/Scrivania/res1.jpg",im)
    boxes = img_as_boxes(im,50,3000)







    nb = len(boxes)
    print("len = "+str(nb))

    
        

    grade = 0
    for a in range(2,10):
        if a*a == nb:
            grade = a
            break
    
    if grade==0:
        return None

    tmp = []

    for box in boxes:
        t = make_16x16(box_as_img(im,box))
        tmp.append({'v':net.predict(t.ravel())[0], 'x':box["x"], 'y':box["y"]})

    res = []
    tmp = sorted(tmp, key=lambda k: k['y'])
    for a in range(0,grade):
        t = tmp[a*grade:(a+1)*grade]
        t = sorted(t, key=lambda k: k['x'])
        for b in range(0,grade):
            res.append(t[b]["v"])

    res = np.array(res).reshape(grade,grade)
    return res

def img_to_matrix_3x3(im,net):
    im = resize_h(im,1024)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    #im = cv2.fastNlMeansDenoisingColored(im,None,3,3,7,21)

    im = zone_clean1(im,3,3,4)
    boxes = img_as_boxes(im,50,3000)
    nb = len(boxes)
    print("len = "+str(nb))

    tmp = []

    if nb==9:
        for box in boxes:
            t = make_16x16(box_as_img(im,box))
            #show_img(t)
            tmp.append({'v':net.predict(t.ravel())[0], 'x':box["x"], 'y':box["y"]})

    
    res = []
    tmp = sorted(tmp, key=lambda k: k['y'])
    for a in range(0,3):
        t = tmp[a*3:(a+1)*3]
        t = sorted(t, key=lambda k: k['x'])
        for b in range(0,3):
            res.append(t[b]["v"])

    res = np.array(res).reshape(3,3)
    return res

def create_network(db):

    imgs = [ele["img"] for ele in db]
    values = [ele["value"] for ele in db]

    train_x, test_x, train_y, test_y = train_test_split(imgs,values)

    print("training data points: {}".format(len(train_y)))
    print("testing data points: {}".format(len(test_y)))


    kVals = range(1, 30, 2)

    # loop over various values of `k` for the k-Nearest Neighbor classifier

    for k in range(1, 30, 2):
            # train the k-Nearest Neighbor classifier with the current value of `k`
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(train_x, train_y)

    model = KNeighborsClassifier(n_neighbors=29)
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
im = zone_clean1(im,3,3,4)


#A = clean_img_to_matrix(im, net)

boxes = img_as_boxes(im,30,100000)

for box in boxes:
    correct_box(im,box,3,3)
#cv2.imwrite("/home/java/Scrivania/c.jpg",im)
boxes = img_as_boxes(im,30,100000)



'''
tmp = []
for box in boxes:
    img = box_as_img(im,box)
    cv2.imwrite("/home/java/Scrivania/res.jpg",img)
    x=0
    try:
        x = net.predict(make_16x16(img).ravel())[0]
        print("prevedo: ",x)
        x = int(raw_input("numero "))
    except:
        print("giusto")
    
    tmp.append({"img":img,"value":x})


fw = open('dbc.pkl', 'wb')
pickle.dump(tmp, fw)
fw.close()
'''

#A = as_square_matrix(im,boxes,net)
A = as_generic_matrix(im,boxes,net,3,3)


print(A)
print("DET = "+str(int(np.linalg.det(A))))
