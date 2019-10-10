import cv2
import math
import csv
import sklearn
import numpy as np



sharp1 = []
sharp2 = []
sharp3 = []

blur1 = []
blur2 = []
blur3 = []

sharpfinal = []

with open('/home/kevin/Retinophy_project/retinopathy_video9/classifySharpBlurVD9(2).csv') as f:
    reader1 = csv.reader(f)
    sharp1 = next(reader1)
    blur1 = next(reader1)

with open('/home/kevin/Retinophy_project/retinopathy_video9/classifySharpBlurVD9(3).csv') as f:
    reader2 = csv.reader(f)
    sharp2 = next(reader2)
    blur2 = next(reader2)

with open('/home/kevin/Retinophy_project/retinopathy_video9/classifySharpBlurVD9(4).csv') as f:
    reader3 = csv.reader(f)
    sharp3 = next(reader3)
    blur3 = next(reader3)

print(len(sharp1))
# print(blur1)
print(len(sharp2))
# print(blur2)
print(len(sharp3))
# print(blur3)
blurfinal = list(set(blur3).intersection(blur1, blur2))

sharpfinal = list(set(sharp3).intersection(sharp1, sharp2))
print(len(sharpfinal))

data = []

data.append(sharpfinal)
data.append(blurfinal)

with open('/home/kevin/Retinophy_project/retinopathy_video9/groundTruthVD9.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(data)
