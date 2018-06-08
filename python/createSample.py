import os

def createPosSample():
    files = os.listdir("E:\\VS Programming\\Object Detection\\opencv_barr_boosted\\ObjectDection\\traincascade\\pos_image")
    name = os.path.split(files[0])
    with open('pos.dat','w+') as f:
        for file in files:
            name = os.path.split(file)
            if name[1] == 'pos.dat':
                continue
            print(file)
            f.write(name[1]+' '+'1 0 0 24 24\n')

def createNegSample():
    files = os.listdir("E:\\VS Programming\\Object Detection\\opencv_barr_boosted\\ObjectDection\\traincascade\\neg_image")
    name = os.path.split(files[0])
    with open('neg.dat','w+') as f:
        for file in files:
            name = os.path.split(file)
            if name[1] == 'pos.dat':
                continue
            f.write("neg_image\\"+name[1]+'\n')



createNegSample()

