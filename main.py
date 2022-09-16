import os 
import time
from helper import *

st=time.time()
#########################
######   TODO   #########
#########################

imageFolder="" ### type the directory of the image folder
cuttedAndRowedImages="" ### type the saving results folder
extension__=".JPG"  ### change the extension if the image is different from .jpg
resizeShape=(320, 240) ### edit the resize shape (default is (320,240))
count=0
#########################
#########################
#########################
for filename in os.listdir(imageFolder):
    if filename.endswith(extension__):
        count+=1 
        img_path=os.path.join(imageFolder, filename)
        image=identifyGreen(img_path,resizeShape) ### first method
        centre=image.shape[1]/2  ### center of columns
        X=image.shape[0] ###   # of rows 
        Y=image.shape[1] ###   # of columns
        image=greyToBin(image) ### second method
        image=remNoise(image)  ### third method 
        image=forthMain(image,X,centre) ### forth method
        image=fifthmain(image,0.5) ### fifth method (rate is given as 0.5 editable)
        image=sixthmain(image) ### sixth method (not completed)
        result=polyfitLines(image) ### extracting lines (not completed fully)
        
        result=result*200 ### for visibility this line can be deleted
        im = Image.fromarray(result)
        im=im.convert('RGB')
        im.save(os.path.join(cuttedAndRowedImages, filename)) ### saving each image in one file


et=time.time()

print("Check out the "+cuttedAndRowedImages+ " folder, you may find {} images with rows with the same name and extension that you give as an input.".format(count))
print("Execution time for the {} images is {} seconds".format(count,et-st))
print("Execution time for per image is {} seconds".format((et-st)/count))
