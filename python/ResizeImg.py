
import cv2 as cv
import os.path
import glob
from PIL import Image

def convertSize(infiles,outfiles,width=24,height=24):
    img=Image.open(infiles)
    try:
       new_img = img.resize((width,height),Image.BILINEAR)
       new_img.save(os.path.join(outfiles,os.path.basename(infiles)))
    except Exception as e:
        print(e)

for files in glob.glob("E:\\Learning Material\\OpenCV\\人脸检测数据库\\Yale大学人脸数据库\\*.bmp"):
    convertSize(files,"E:\\Learning Material\\OpenCV\\人脸检测数据库\\24x24")

