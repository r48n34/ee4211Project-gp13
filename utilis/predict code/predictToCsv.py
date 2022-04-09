import torch
import os
import pandas as pd
import numpy as np 
import PIL

# model location
model = torch.hub.load('ultralytics/yolov5', 'custom', path="G:\\yolov5Play\\yolov5\\birdPt\\best.pt")

imgDir = "G:\\Data\\Test"
imgArr = os.listdir(imgDir)

# imgArr to proper img with path
imgs = [ os.path.join(imgDir,i) for i in imgArr]

model.conf = 0.20  # NMS confidence threshold
model.iou = 0.30  # NMS IoU threshold

# transfer bbox to coco format
def calBoxPt (xmin,ymin, xmax, ymax, width, height):
    x = xmin / width
    y = ymin / height

    w = (xmax - xmin) / width
    h = (ymax - ymin) / height

    return [x,y,w,h]

finalArr = []
for ind, img in enumerate(imgs):
    image = PIL.Image.open(img)
    width, height = image.size

    results = model(img) # predict result
    res = results.pandas().xyxy[0] # predict result to bbox arr

    if len(res.confidence) >= 1 : # have predict result box

        boxPt = calBoxPt(res.xmin[0], res.ymin[0], res.xmax[0], res.ymax[0], width, height)
        resultStr = str(1) + " " + str(res.confidence[0]) + " " + " ".join([str(st) for st in boxPt]) # string format

        finalArr.append([ imgArr[ind] ,  resultStr ])
    else:
        finalArr.append([ "" ])


print(len(finalArr))
finalArr = np.array(finalArr)
df = pd.DataFrame({'ImageId': finalArr[:, 0], 'PredictionString': finalArr[:, 1]})
df.to_csv("output.csv" , index=False)
