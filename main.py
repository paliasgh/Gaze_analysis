#region
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import pchip
import Image_processing_func
import pyautogui
from math import atan2
from math import pi
from math import degrees
from math import sqrt
import pandas as pd
import pickle
import math
from os import listdir
from os.path import isfile, join

def ellipse_eqn_in(x,y,cnetre_point,maj_ax,min_ax,rot_ang):
    xa = x - cnetre_point[0]
    ya = y - cnetre_point[1]
    term1 = math.pow(xa*math.cos(math.radians(rot_ang))+ya*math.sin(math.radians(rot_ang)),2)
    term2 = math.pow(xa*math.sin(math.radians(rot_ang))-ya*math.cos(math.radians(rot_ang)),2)
    if (term1/math.pow(maj_ax,2))+(term2/math.pow(min_ax,2))<1:
        return True
    else:
        return False

def save_mouse(event, x, y, flags, param):
    global mouseX, mouseY, click_x, click_y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        click_x = mouseX
        click_y = mouseY
        pyautogui.press("enter")
    return True
#endregion
folder = 'Test_5/'
files = [f for f in listdir(folder) if isfile(join(folder, f))]
file = 'Test/test1.xlsx'
file = folder + files[0]
# Load video and gaze file
#cap = cv2.VideoCapture('Test/1.mp4')
df = pd.read_excel(file, skiprows=[1], skipfooter=1)
print("data is loaded")

# Wait
participant = df.iat[1, 5]
session = str(df.iat[1, 6])[-1]
# Values from Analyzer
#first_frame_timestamp = 430
#first_frame_timestamp_num = 4
# Pink sponge
dark_limit_p = np.array([255, 255, 255])
light_limit_p = np.array([150, 100, 180])
# Yellow sponge
dark_limit_p = np.array([255, 255, 255])
light_limit_p = np.array([0,120,155])

### Process
#region

media_file = df['Recording media name']
media_file = media_file.tolist()
index_first_frame = -1
video_name = []
for i in media_file:
    index_first_frame = index_first_frame + 1
    if str(i) != "nan":
        video_name = i
        break
video_name = video_name.replace("/", "_")
video_name = video_name.replace("+", "-")
#video_name = "apdZYHEYdcZj8W4YME6gEA==.mp4" # for vidhi s 1
cap = cv2.VideoCapture('Media/'+video_name)
# Find video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# The frame that each row should correspond to
df['Video frame th'] = df['Recording timestamp']*fps/1000

# Keep only eye tracker rows + tasks
is_ = df['Event']=="SyncPortOutLow"
is_ = is_.tolist()
is_ = np.where(is_)[0]
df = df.drop(is_).reset_index(drop=True)
is_ = df['Event']=="SyncPortOutHigh"
is_ = is_.tolist()
is_ = np.where(is_)[0]
df = df.drop(is_).reset_index(drop=True)

# Find the index of the video first frame
timestamps = df['Recording timestamp']
#timestamps = timestamps.tolist()
#index_first_frame = timestamps.index(first_frame_timestamp)

# The frame that each row actually corresponds to
#first_frame = df.iat[index_first_frame, 17]
#df['Video frame'] = np.trunc(df['Video frame th']) + 1 - np.trunc(first_frame)

# The other way!
first_frame = df.iat[index_first_frame, 17]
df['Video frame'] = np.trunc(df['Video frame th']) + 1 - np.trunc(first_frame)

# Find the index of first frame of Task 1
events = df['Event']
events = events.tolist()
index_t1_s = events.index("Task 1 - Start")
index_t1_e = events.index("Task 1 - End")

# Save gaze location and frames
frames_t1 = list(range(int(df.iat[index_t1_s, 18]), int(df.iat[index_t1_e, 18]+1)))
frames = df['Video frame']
frames_th = df['Video frame th']
gaze_xs = df['Gaze point X']
gaze_ys = df['Gaze point Y']
gaze_type = df['Eye movement type']
frames = frames.tolist()
#endregion
print(participant)

scale_factor_maj = 1.9
scale_factor_min = 1.5
### Record
#region
click_x = 0
click_y = 0
cnetre_points_x = []
cnetre_points_y = []
maj_axs = []
min_axs = []
rot_angs = []
clicks_frame = []
clicks_frame_video = []

count = -1
for c in frames_t1:
    ret, frame = cap.read()
    cap.set(1, c)
    if ret == True:
        count = count + 1
        if count == 0:  # Skip the first frame
            print("Skip")
            continue
        assert isinstance(count, object)
        if count % 20 == 0:  # Every 20 frames
            clicks_frame.append(count)
            clicks_frame_video.append(c)
            box = []
            img_ellipse, cnetre_point, maj_ax, min_ax, rot_ang = Image_processing_func.find_sponge(frame,
                                                                                                   scale_factor_maj, scale_factor_min
                                                                                                   , dark_limit_p, light_limit_p)
            cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
            cv2.imshow('Frame', img_ellipse)
            k = cv2.waitKey(0)
            # cv2.setMouseCallback('Frame', Image_processing_func.send_key)
            if k == 113:  # q key to stop
                break
            elif k == 97:  # Press a
                print("Accepted")
                cnetre_points_x.append(cnetre_point[0])
                cnetre_points_y.append(cnetre_point[1])
                maj_axs.append(maj_ax)
                min_axs.append(min_ax)
                rot_angs.append(rot_ang)

                # When accepted, next one will be the last one!
                # Need to save box (in every frame?) to later edit
            else:
                print("Save a new one")
                for i in range(4):
                    # cv2.waitKey(0)
                    cv2.setMouseCallback('Frame', save_mouse)
                    cv2.waitKey(0)
                    print("Clicks:", click_x, click_y)
                    box.append([click_x, click_y])
                box = np.array(box)
                rot_ang = degrees(atan2((box[2, 1] - box[1, 1]), (box[2, 0] - box[1, 0])))  # =a!
                maj_ax = int(sqrt((box[2, 1] - box[1, 1]) * (box[2, 1] - box[1, 1]) + (box[2, 0] - box[1, 0]) * (
                        box[2, 0] - box[1, 0])) / 2)
                min_ax = int(sqrt((box[0, 1] - box[1, 1]) * (box[0, 1] - box[1, 1]) + (box[0, 0] - box[1, 0]) * (
                        box[0, 0] - box[1, 0])) / 2)
                cnetre_point = (int(np.mean(box[:, 0])), int(np.mean(box[:, 1])))
                maj_ax_mod = int(maj_ax * scale_factor_maj)
                min_ax_mod = int(maj_ax * scale_factor_min)
                new_img_ellipse = cv2.ellipse(img_ellipse.copy(), cnetre_point, (maj_ax_mod, min_ax_mod), rot_ang, 0,
                                              360, (0, 255, 0), 3)
                new_img_box = cv2.drawContours(new_img_ellipse.copy(), [box], 0, (0, 255, 0), 3)
                cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
                cv2.imshow('Frame', new_img_box)
                k2 = cv2.waitKey(0)
                if k2 == 97:    # Press a
                    print("New one accepted")
                    cnetre_points_x.append(cnetre_point[0])
                    cnetre_points_y.append(cnetre_point[1])
                    maj_axs.append(maj_ax)
                    min_axs.append(min_ax)
                    rot_angs.append(rot_ang)
                    print(box)
                else:
                    print("Save a new one, again!")
                    box = []
                    for i in range(4):
                        # cv2.waitKey(0)
                        cv2.setMouseCallback('Frame', save_mouse)
                        cv2.waitKey(0)
                        print("Clicks:", click_x, click_y)
                        box.append([click_x, click_y])
                    box = np.array(box)
                    rot_ang = degrees(atan2((box[2, 1] - box[1, 1]), (box[2, 0] - box[1, 0])))  # =a!
                    maj_ax = int(sqrt((box[2, 1] - box[1, 1]) * (box[2, 1] - box[1, 1]) + (box[2, 0] - box[1, 0]) * (
                            box[2, 0] - box[1, 0])) / 2)
                    min_ax = int(sqrt((box[0, 1] - box[1, 1]) * (box[0, 1] - box[1, 1]) + (box[0, 0] - box[1, 0]) * (
                            box[0, 0] - box[1, 0])) / 2)
                    cnetre_point = (int(np.mean(box[:, 0])), int(np.mean(box[:, 1])))
                    maj_ax_mod = int(maj_ax * scale_factor_maj)
                    min_ax_mod = int(maj_ax * scale_factor_min)
                    new_img_ellipse = cv2.ellipse(img_ellipse.copy(), cnetre_point, (maj_ax_mod, min_ax_mod), rot_ang,
                                                  0, 360, (255, 0, 0), 3)
                    new_img_box = cv2.drawContours(new_img_ellipse.copy(), [box], 0, (255, 0, 0), 3)
                    cnetre_points_x.append(cnetre_point[0])
                    cnetre_points_y.append(cnetre_point[1])
                    maj_axs.append(maj_ax)
                    min_axs.append(min_ax)
                    rot_angs.append(rot_ang)
                    print(box)
                    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
                    cv2.imshow('Frame', new_img_box)
                    cv2.waitKey(1000)
#        else:
            #cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
            #cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
#cap.release()
cv2.destroyAllWindows()
#endregion


# If need to change:
scale_factor_maj = 2
scale_factor_min = 1.6

### Calculate:
#region
# Fit
cnetre_points_x_spl = UnivariateSpline(clicks_frame, cnetre_points_x)
cnetre_points_y_spl = UnivariateSpline(clicks_frame, cnetre_points_y)
maj_axs_mod = [i * scale_factor_maj for i in maj_axs]
min_axs_mod = [i * scale_factor_min for i in min_axs]
maj_axs_spl = UnivariateSpline(clicks_frame, maj_axs_mod)
min_axs_spl = UnivariateSpline(clicks_frame, min_axs_mod)
rot_angs_spl = UnivariateSpline(clicks_frame, rot_angs)

df["Location"] = np.nan
offset = clicks_frame_video[0] - clicks_frame[0]
count = 0
count_in = 0
for i in range(index_t1_s+1,index_t1_e):
    val = frames_th[i] - offset
    try:
        if (ellipse_eqn_in(int(gaze_xs[i]), int(gaze_ys[i]),
                   (int(cnetre_points_x_spl(val)), int(cnetre_points_y_spl(val))),
                   int(maj_axs_spl(val)), int(min_axs_spl(val)), int(rot_angs_spl(val)))):
            if gaze_type[i] == 'Fixation':
                df.at[i, "Location"] = "In"
                count_in = count_in + 1
            else:
                df.at[i, "Location"] = "In, no fixation"
        else:
            df.at[i, "Location"] = "Out"
        count = count + 1
    except:
        df.at[i, "Location"] = "No data"
print(participant, " s", session)
print("Total gazes:", count)
print("Total at_end_effector gazes:", count_in)
print("At_end_effector gaze:", count_in/count*100, "%")
#endregion

### Play
#region
image_ell = np.full((1080, 1920, 3), (255,0,255), dtype=np.uint8)
count = -1
for c in frames_t1:
    ret, frame = cap.read()
    cap.set(1, c)
    if ret == True:
        count = count + 1
        if count == 0:  # Skip the first frame
            print("Skip")
            continue
        try:
            index_df = frames.index(c)
            if(gaze_type[index_df] == 'Fixation'):   #Green
                image = cv2.circle(frame, (int(gaze_xs[index_df]), int(gaze_ys[index_df])), radius=20, color=(0, 255, 0),
                               thickness=-1)
            elif(gaze_type[index_df] == 'Saccade'):  #Blue
                image = cv2.circle(frame, (int(gaze_xs[index_df]), int(gaze_ys[index_df])), radius=20, color=(255, 0, 0),
                               thickness=-1)
            else:                                    #Red
                image = cv2.circle(frame, (int(gaze_xs[index_df]), int(gaze_ys[index_df])), radius=20, color=(0, 0, 255),
                               thickness=-1)
        except:
            print("No data in frame", c)
        assert isinstance(count, object)
        try:                                         #Green: In
            if ellipse_eqn_in(int(gaze_xs[index_df]), int(gaze_ys[index_df]), (int(cnetre_points_x_spl(count)), int(cnetre_points_y_spl(count))),
                                int(maj_axs_spl(count)), int(min_axs_spl(count)), int(rot_angs_spl(count))):
                image_ell = cv2.ellipse(frame.copy(), (int(cnetre_points_x_spl(count)), int(cnetre_points_y_spl(count))),
                                    (int(maj_axs_spl(count)), int(min_axs_spl(count))), (int(rot_angs_spl(count))), 0, 360, (0, 255, 0), 3)
            else:                                    #Red: Out
                image_ell = cv2.ellipse(frame.copy(), (int(cnetre_points_x_spl(count)), int(cnetre_points_y_spl(count))),
                                    (int(maj_axs_spl(count)), int(min_axs_spl(count))), (int(rot_angs_spl(count))), 0, 360, (0, 0, 255), 3)
        except:
            print("No gaze data", c)

        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.imshow('Frame', image_ell)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;
    else:
        break
#cap.release()
cv2.destroyAllWindows()
# endregion

# Save
with open('Outputs/' + file[:-5] +'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([clicks_frame, clicks_frame_video, cnetre_points_x, cnetre_points_y, maj_axs, min_axs, rot_angs, scale_factor_maj, scale_factor_min], f)
df.to_excel('Outputs/' + file[:-5] + " - output.xlsx")
print("Saved")


# Load
with open('Output/' + file[:-5] +'.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    clicks_frame, clicks_frame_video, cnetre_points_x, cnetre_points_y, maj_axs, min_axs, rot_angs, scale_factor_maj, scale_factor_min= pickle.load(f)