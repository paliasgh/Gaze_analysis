#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:05:26 2022

@author: pourya
"""
from math import atan2
from math import pi
from math import degrees
from math import sqrt
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np

def onclick(event):
   print([event.xdata, event.ydata])
   
image = cv2.imread("Random Gaze Pics/7.png")
height, width = image.shape[:2]


fig,ax=plt.subplots(figsize=(4, 4), dpi=300)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#plt.title('Face image')
plt.axis('off')
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()



import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10,10)
y = x**2

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,y)

coords = []

def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print (ix, iy)

    global coords
    coords.append((ix, iy))

    if len(coords) == 2:
        fig.canvas.mpl_disconnect(cid)

    return coords
cid = fig.canvas.mpl_connect('button_press_event', onclick)
# %%
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
ch = cv2.split(image)

plt.imshow(cv2.cvtColor(ch[2], cv2.COLOR_BGR2RGB))
#plt.title('Face image')
plt.axis('off')
plt.show()

#region Process
img_blur = cv2.GaussianBlur(image, (3,3), 0)
edges = cv2.Canny(image=img_blur, threshold1=5, threshold2=100) # Canny Edge Detection
plt.figure(figsize=(3, 4), dpi=300)
plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
#endregion

##### region Process
from math import atan2
from math import pi
from math import degrees
from math import sqrt
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
# Read the image
image = cv2.imread("Random Gaze Pics/19.png")

# Apply color mask
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
dark_blue = np.array([70, 130, 230])
light_blue = np.array([25, 70, 50])
mask = cv2.inRange(hsv, light_blue, dark_blue)

output = cv2.bitwise_and(image,image, mask= mask)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
#endregion

# Apply opening filter
mask_new = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(18,18)))
plt.imshow(cv2.cvtColor(mask_new, cv2.COLOR_BGR2RGB))

# x, y, w, h = cv2.boundingRect(mask_new)

# Find and merge contours
contours,_ = cv2.findContours(mask_new.copy(), 1, 1)
list_of_pts = []
for ctr in contours:
    list_of_pts += [pt[0] for pt in ctr]

class clockwise_angle_and_distance():
    def __init__(self, origin):
        self.origin = origin

    def __call__(self, point, refvec=[0, 1]):
        if self.origin is None:
            raise NameError("clockwise sorting needs an origin. Please set origin.")
        vector = [point[0] - self.origin[0], point[1] - self.origin[1]]
        lenvector = np.linalg.norm(vector[0] - vector[1])
        if lenvector == 0:
            return -pi, 0
        normalized = [vector[0] / lenvector, vector[1] / lenvector]
        dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]  # x1*x2 + y1*y2
        diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]  # x1*y2 - y1*x2
        angle = atan2(diffprod, dotprod)
        if angle < 0:
            return 2 * pi + angle, lenvector
        return angle, lenvector

center_pt = np.array(list_of_pts).mean(axis=0)
clock_ang_dist = clockwise_angle_and_distance(center_pt)
list_of_pts = sorted(list_of_pts, key=clock_ang_dist)
ctr = np.array(list_of_pts).reshape((-1,1,2)).astype(np.int32)
ctr = cv2.convexHull(ctr) # done.

# Fit rotated rectangle
rect = cv2.minAreaRect(mask)
(x,y),(w,h), a = rect
box = cv2.boxPoints(rect)
box = np.int0(box)
rect2 = cv2.drawContours(image.copy(),[box],0,(0,0,255),3)
###

plt.imshow(cv2.cvtColor(rect2, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
#endregion

# Fit ellipse
rot_ang = degrees(atan2((box[2,1]-box[1,1]),(box[2,0]-box[1,0]))) # =a!
scale_factor = 1.3
maj_ax = int(sqrt((box[2,1]-box[1,1])*(box[2,1]-box[1,1])+(box[2,0]-box[1,0])*(box[2,0]-box[1,0]))/2*scale_factor)
min_ax = int(sqrt((box[0,1]-box[1,1])*(box[0,1]-box[1,1])+(box[0,0]-box[1,0])*(box[0,0]-box[1,0]))/2*scale_factor)
cnetre_point = (int(np.mean(box[:, 0])), int(np.mean(box[:, 1])))
rect3 = cv2.ellipse(rect2.copy(), cnetre_point, (maj_ax, min_ax), rot_ang, 0,360, 255, 3)
###

# See the masked image
output = cv2.bitwise_and(image,image, mask= mask_new)
###

def ellipse_eqn_in(x,y,cnetre_point,maj_ax,min_ax,rot_ang):
    xa = x - cnetre_point[0]
    ya = y - cnetre_point[1]
    term1 = math.pow(xa*math.cos(math.radians(rot_ang))+ya*math.sin(math.radians(rot_ang)),2)
    term2 = math.pow(xa*math.sin(math.radians(rot_ang))-ya*math.cos(math.radians(rot_ang)),2)
    if (term1/math.pow(maj_ax,2))+(term2/math.pow(min_ax,2))<1:
        return True
    else:
        return False

# Test in or out
test_x = 920
test_y = 550
rect4 = cv2.circle(rect3.copy(), (test_x,test_y), radius=10, color=(0, 0, 255), thickness=-1)
print(ellipse_eqn_in(test_x,test_y,cnetre_point,maj_ax,min_ax,rot_ang))
###

plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
#endregion

#region Show
fig,ax=plt.subplots(figsize=(4, 4), dpi=300)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
#endregion

















import cv2
import numpy as np

cap = cv2.VideoCapture('Sample videos/1t1.mp4')
success,image = cap.read()
count = 0
frames = []
frames_cv = []

while success:
    frames.append(image)
    frames_cv.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    success,image = cap.read()
    print('Read frame: ' + str(count))
    count += 1
    
# %%    

# def draw_circle(event,x,y,flags,param):
#     global mouseX,mouseY
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         cv2.circle(img,(x,y),100,(255,0,0),-1)
#         mouseX,mouseY = x,y
        
# img = np.zeros((512,512,3), np.uint8)
# cv2.namedWindow('image')
# cv2.setMouseCallback('image',draw_circle)        

count = 0        
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    cv2.imshow('Frame',frame)
    count = count + 1
    print(count)
    if count%10 == 0:
        cv2.waitKey(1000)
    if cv2.waitKeqy(40) & 0xFF == ord('q'):
        break;
  else: 
      break;

cap.release()
cv2.destroyAllWindows()

# %% Record

import cv2
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(3, 4), dpi=20)
count = 0

def onclick(event):
   print([event.xdata, event.ydata])

for f in frames_cv:
    count = count + 1
    # if count%20 == 0:
    #     fig.canvas.mpl_connect('button_press_event', onclick)
    #     input()
    plt.imshow(f[::7,::7])
    plt.axis('off')
    plt.show()   
    
    

# %%

plt.figure(figsize=(3, 4), dpi=300)
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
  else: 
    break

plt.figure(figsize=(3, 4), dpi=300)
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#plt.title('Face image')
plt.axis('off')
plt.show()

