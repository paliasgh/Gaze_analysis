import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import atan2
from math import pi
from math import degrees
from math import sqrt
import math

def send_key(event, x, y, flags, param):
    global mouseX, mouseY, click_x, click_y
    if event == cv2.EVENT_LBUTTONDOWN:
        if click_x == 0:
            mouseX, mouseY = x, y
            click_x = mouseX
            click_y = mouseY
        pyautogui.press("enter")
    return True

def find_sponge(image, scale_factor_maj, scale_factor_min, dark_limit, light_limit):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, light_limit, dark_limit)

    # Apply opening filter
    mask_new = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18)))

    # x, y, w, h = cv2.boundingRect(mask_new)

    # Find and merge contours
    contours, _ = cv2.findContours(mask_new.copy(), 1, 1)
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

    try:
        center_pt = np.array(list_of_pts).mean(axis=0)
        clock_ang_dist = clockwise_angle_and_distance(center_pt)
        list_of_pts = sorted(list_of_pts, key=clock_ang_dist)
        ctr = np.array(list_of_pts).reshape((-1, 1, 2)).astype(np.int32)
        ctr = cv2.convexHull(ctr)  # done.

        # Fit rotated rectangle
        rect = cv2.minAreaRect(ctr)
        (x, y), (w, h), a = rect
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        rect2 = cv2.drawContours(image.copy(), [box], 0, (0, 0, 255), 3)
        ###

        # Fit ellipse
        rot_ang = degrees(atan2((box[2, 1] - box[1, 1]), (box[2, 0] - box[1, 0])))  # =a!
        maj_ax = int(sqrt((box[2, 1] - box[1, 1]) * (box[2, 1] - box[1, 1]) + (box[2, 0] - box[1, 0]) * (
                    box[2, 0] - box[1, 0])) / 2 )
        min_ax = int(sqrt((box[0, 1] - box[1, 1]) * (box[0, 1] - box[1, 1]) + (box[0, 0] - box[1, 0]) * (
                    box[0, 0] - box[1, 0])) / 2 )
        maj_ax_mod = int(maj_ax * scale_factor_maj)
        min_ax_mod = int(maj_ax * scale_factor_min)
        cnetre_point = (int(np.mean(box[:, 0])), int(np.mean(box[:, 1])))
        rect3 = cv2.ellipse(rect2.copy(), cnetre_point, (maj_ax_mod, min_ax_mod), rot_ang, 0,
                            360, (0, 0, 255), 3)
    except:
        maj_ax = 100
        min_ax = 100
        cnetre_point = (500, 500)
        rot_ang = 0
        rect3 = cv2.ellipse(image.copy(), cnetre_point, (maj_ax, min_ax), rot_ang, 0,
                            360, (0, 0, 255), 3)
        print("Unable to locate the sponge")
    ###

    # See the masked image
    output = cv2.bitwise_and(image, image, mask=mask_new)
    ###



    return(rect3,cnetre_point,maj_ax,min_ax,rot_ang)



def find_board(image, image_show, dark_limit, light_limit):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, light_limit, dark_limit)

    # Apply opening filter
    mask_new = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)))

    # x, y, w, h = cv2.boundingRect(mask_new)

    # Find and merge contours
    contours, _ = cv2.findContours(mask_new.copy(), 1, 1)
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

    try:
        center_pt = np.array(list_of_pts).mean(axis=0)
        clock_ang_dist = clockwise_angle_and_distance(center_pt)
        list_of_pts = sorted(list_of_pts, key=clock_ang_dist)
        ctr = np.array(list_of_pts).reshape((-1, 1, 2)).astype(np.int32)
        ctr = cv2.convexHull(ctr)  # done.

        # Fit rotated rectangle
        rect = cv2.minAreaRect(ctr)
        (x, y), (w, h), a = rect
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        #rect2 = cv2.drawContours(image_show.copy(), [box], 0, (0, 0, 255), 3)
        ###
    except:

        #rect2 = cv2.drawContours(image.copy(), [box], 0, (0, 0, 255), 3)

        print("Unable to locate the board")
    ###

    return(box)

#image_ex = cv2.imread("Random Gaze Pics/12.png")

#dark_limit_p = np.array([255, 255, 255])
#light_limit_p = np.array([150, 100, 180])
#img_ellipse, cnetre_point, maj_ax, min_ax, rot_ang = find_sponge(image_ex,2.2,1.2,dark_limit_p, light_limit_p)
#plt.imshow(cv2.cvtColor(img_ellipse, cv2.COLOR_BGR2RGB))
#plt.axis('off')
#plt.show()

#dark_limit_p = np.array([255,255,255])
#light_limit_p = np.array([50,135,10])
#img_rec = find_board(image_ex, image_ex,dark_limit_p, light_limit_p)
#plt.imshow(cv2.cvtColor(img_rec, cv2.COLOR_BGR2RGB))
#plt.axis('off')
#plt.show()

def ellipse_eqn_in(x,y,cnetre_point,maj_ax,min_ax,rot_ang):
    xa = x - cnetre_point[0]
    ya = y - cnetre_point[1]
    term1 = math.pow(xa*math.cos(math.radians(rot_ang))+ya*math.sin(math.radians(rot_ang)),2)
    term2 = math.pow(xa*math.sin(math.radians(rot_ang))-ya*math.cos(math.radians(rot_ang)),2)
    if (term1/math.pow(maj_ax,2))+(term2/math.pow(min_ax,2))<1:
        return True
    else:
        return False

