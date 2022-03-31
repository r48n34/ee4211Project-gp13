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

results = model(imgs)
results.print() 
results.save()  
