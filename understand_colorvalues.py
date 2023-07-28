import cv2
cv2.imshow('',map)
dict1 = {}
for i in map[:,10]:
    t=tuple(i)
    if t in dict1:
        dict1[t]+=1
    else:
        dict1[t] =1
dict1