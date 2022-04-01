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

        #boxPt = calBoxPt(res.xmin[0], res.ymin[0], res.xmax[0], res.ymax[0], width, height)
        #resultStr = str(1) + " " + str(res.confidence[0]) + " " + " ".join([str(st) for st in boxPt]) # string format

        finalArr.append([ imgArr[ind] , int(res.xmin[0]), int(res.ymin[0]), int(res.xmax[0]), int(res.ymax[0])  ])
        #finalArr.append([ imgArr[ind] , X_min, Y_min, X_max, Y_max  ])
    else:
        finalArr.append([ imgArr[ind], 0,0,0,0 ])


print(len(finalArr))
finalArr = np.array(finalArr)
df = pd.DataFrame({
    'ImageId': finalArr[:, 0],
    'X_min': finalArr[:, 1],
    'Y_min': finalArr[:, 2],
    'X_max': finalArr[:, 3],
    'Y_max': finalArr[:, 4]
})

df.to_csv("outputNew2.csv" , index=False)
