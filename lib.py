import cv2
import numpy as np


def show_img(im):
    cv2.imshow("image",im)
    cv2.waitKey()
    cv2.destroyAllWindows()

def zone_clean(im,w0,h0,lvl):
    h = im.shape[0]
    w = im.shape[1]

    a = h/h0
    b = w/w0
    
    img = 0
    

    for x in range(0,h0):

        row = 0

        for y in range(0,w0):
            imz = im[a*x:a*(x+1), b*y:b*(y+1)]
            blur = int(cv2.Laplacian(imz, cv2.CV_64F).var())
            v = round(math.log(blur,2))+lvl
            print(v)

            imz = cv2.cvtColor(imz, cv2.COLOR_BGR2GRAY)
            imz = cv2.adaptiveThreshold(imz,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,v)
            #block.append(np.array(imz))
            if y==0:
                row = np.array(imz)
            else:
                row = np.concatenate((row,imz),axis=1)

        if x==0:
            img = row
        else:
            img = np.concatenate((img,row),axis=0)

    return img

def zone_clean1(im,w0,h0,q):
    h = im.shape[0]
    w = im.shape[1]

    a = h/h0
    b = w/w0
    
    img = 0
    

    for x in range(0,h0):

        row = 0

        for y in range(0,w0):
            imz = im[a*x:a*(x+1), b*y:b*(y+1)]
            imz = cv2.adaptiveThreshold(imz,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,33,16)
            #block.append(np.array(imz))
            if y==0:
                row = np.array(imz)
            else:
                row = np.concatenate((row,imz),axis=1)

        if x==0:
            img = row
        else:
            img = np.concatenate((img,row),axis=0)

    return img

def img_as_boxes(img, min_area, max_area):
    _, ctrs, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    tmp = []
    for ctr in ctrs:

        area = cv2.contourArea(ctr)
        if (area>min_area and area<max_area):
            x,y,w,h = cv2.boundingRect(ctr)
            tmp.append({'x':x,'y':y,'w':w,'h':h})

    return tmp

def box_as_img(img, box):
    #print(box)
    #print(box["y"],":",box["y"]+box["h"],":",box["x"],":",box["x"]+box["w"])
    return img[box["y"]:box["y"]+box["h"],box["x"]:box["x"]+box["w"]]

def resize_h(im, a):
    h = im.shape[0]
    w = im.shape[1]

    rate = float(a)/h
    return cv2.resize(im, (0,0), fx=rate, fy=rate)

def correct_box(im, box,ori,ver):
    h = box["h"]
    w = box["w"]
    x = box["x"]
    y = box["y"]

    if ori>0:
        s = 0
        j = 100
        for ty in range(0,h):
            for tx in range(0,w):
                #print(x)
                if im[y+ty,x+tx]>0:
                    #print("dist:",j)
                    if j<=ori and j>0:
                        for a in range(tx-j,tx):
                            if a<0:
                                a=0
                            im[y+ty,x+a] = 255
                            #print("riempo:",x-a)
                    j=0
                else:
                    j+=1
            j=100

    if ver>0:
        s = 0
        j = 100
        for tx in range(0,w):
            for ty in range(0,h):
                #print(x)
                if im[y+ty,x+tx]>0:
                    #print("dist:",j)
                    if j<=ver and j>0:
                        for a in range(ty-j,ty):
                            if a<0:
                                a=0
                            im[y+a,x+tx] = 255
                            #print("riempo:",x-a)
                    j=0
                else:
                    j+=1
            j=100     

def make_16x16(img):

    h = img.shape[0]
    w = img.shape[1]


    if h>w:
        rate = 16.0/h

        img = cv2.resize(img, (0,0), fx=rate, fy=rate)
        h = img.shape[0]
        w = img.shape[1]

        x = (16-w)/2

        s = np.zeros([16, x], dtype=np.uint8)
        img = np.concatenate((s,img), axis=1)
        s = np.zeros([16, (16-w)-x], dtype=np.uint8)
        img = np.concatenate((img,s), axis=1)
    else:
        rate = 16.0/w

        img = cv2.resize(img, (0,0), fx=rate, fy=rate)
        h = img.shape[0]
        w = img.shape[1]

        x = (16-h)/2

        s = np.zeros([x, 16], dtype=np.uint8)
        img = np.concatenate((s,img), axis=0)
        s = np.zeros([(16-h)-x, 16], dtype=np.uint8)
        img = np.concatenate((img,s), axis=0)

    return img

def correct(im, ori, ver):
    h = im.shape[0]
    w = im.shape[1]

    if ori>0:
        s = 0
        j = 100
        for y in range(0,h):
            for x in range(0,w):
                #print(x)
                if im[y,x]>0:
                    #print("dist:",j)
                    if j<=ori and j>0:
                        for a in range(x-j,x):
                            if a<0:
                                a=0
                            im[y,a] = 255
                            #print("riempo:",x-a)
                    j=0
                else:
                    j+=1
            j=100


    if ver>0:
        s = 0
        j = 100
        for x in range(0,w):
            for y in range(0,h):
                #print(x)
                if im[y,x]>0:
                    #print("dist:",j)
                    if j<=ver and j>0:
                        for a in range(y-j,y):
                            if a<0:
                                a=0
                            im[a,x] = 255
                            #print("riempo:",x-a)
                    j=0
                else:
                    j+=1
            j=100      
    
   
    return im

def as_generic_matrix(im,boxes,net,w,h):
    boxes = sorted(boxes, key=lambda k: k['y'])
    
    '''
    for box in boxes:
        show_img(box_as_img(im,box))
    '''

    tmp = []
    x = False
    for box in boxes:
        x = True
        for c in range(0,len(tmp)):
            if abs(tmp[c][len(tmp[c])-1]["x"]-box["x"])<30 and abs(tmp[c][0]["y"]-box["y"])<10:
                tmp[c].append(box)
                x = False
                break
        if x:
            tmp.append([box])

    
    for c in range(0,len(tmp)):
        if len(tmp[c])>1:
            tmp[c] = sorted(tmp[c], key=lambda k: k['x'])
    for a in range(0,h):
        tmp[a*w:(a+1)*w] = sorted(tmp[a*w:(a+1)*w], key=lambda k: k[0]['x'])

    tmp = np.array(tmp).reshape(h,w)
    A = np.zeros([h,w])
    for a in range(0,h):
        for b in range(0,w):
            num = 0
            arr = tmp[a][b]
            if not isinstance(arr,list):
                arr = [arr]
            
            c=len(arr)-1
            for ele in arr:
                print(">>",ele)
                img = make_16x16(box_as_img(im,ele))
                v = net.predict(img.ravel())
                num += int(v*pow(10,c))
                c-=1
            A[a,b]=num
                
    return A

#correct bug box_as_img as in as_generic_matrix
def as_square_matrix(im,boxes,net):
    boxes = sorted(boxes, key=lambda k: k['y'])
    
    '''
    for box in boxes:
        show_img(box_as_img(im,box))
    '''

    tmp = []
    x = False
    for box in boxes:
        x = True
        for c in range(0,len(tmp)):
            if abs(tmp[c][len(tmp[c])-1]["x"]-box["x"])<40 and abs(tmp[c][0]["y"]-box["y"])<10:
                tmp[c].append(box)
                x = False
                break
        if x:
            tmp.append([box])
    


    print(len(tmp))
    grade = 0
    for a in range(2,10):
        if a*a == len(tmp):
            grade = a
            break

    
    if grade==0:
        return None


    #fase di ordine
    for c in range(0,len(tmp)):
        if len(tmp[c])>1:
            tmp[c] = sorted(tmp[c], key=lambda k: k['x'])
    for a in range(0,grade):
        tmp[a*3:(a+1)*3] = sorted(tmp[a*3:(a+1)*3], key=lambda k: k[0]['x'])

    
    #print(tmp)
    tmp = np.array(tmp).reshape(grade,grade)
    A = np.zeros([grade,grade])


    for a in range(0,grade):
        for b in range(0,grade):
            num = 0
            arr = tmp[a][b]
            c=len(arr)-1
            for ele in arr:
                #print(">>",ele)
                img = make_16x16(box_as_img(im,ele))
                v = net.predict(img.ravel())
                num += int(v*pow(10,c))
                c-=1
            A[a,b]=num
                
    return A





        
    

