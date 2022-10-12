# region
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
import imutils
from imutils.video import FPS
def ellipse_eqn_in(x, y, cnetre_point, maj_ax, min_ax, rot_ang):
    xa = x - cnetre_point[0]
    ya = y - cnetre_point[1]
    term1 = math.pow(xa * math.cos(math.radians(rot_ang)) + ya * math.sin(math.radians(rot_ang)), 2)
    term2 = math.pow(xa * math.sin(math.radians(rot_ang)) - ya * math.cos(math.radians(rot_ang)), 2)
    if (term1 / math.pow(maj_ax, 2)) + (term2 / math.pow(min_ax, 2)) < 1:
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

# endregion

Task = '1'
folder = 'Task' + Task + '/'
files = [f for f in listdir(folder) if isfile(join(folder, f))]
file = folder + files[64]
df = pd.read_excel(file, skiprows=[1], skipfooter=1)
print("data is loaded")
### Process
# region
participant = df.iat[1, 5]
session = str(df.iat[1, 6])[-1]

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
if (participant == "Vidhi" and session == '1'):
    video_name = "apdZYHEYdcZj8W4YME6gEA==.mp4"  # for vidhi s 1
cap = cv2.VideoCapture('Media/' + video_name)
# Find video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# The frame that each row should correspond to
df['Video frame th'] = df['Recording timestamp'] * fps / 1000

# Keep only eye tracker rows + tasks
is_ = df['Event'] == "SyncPortOutLow"
is_ = is_.tolist()
is_ = np.where(is_)[0]
df = df.drop(is_).reset_index(drop=True)
is_ = df['Event'] == "SyncPortOutHigh"
is_ = is_.tolist()
is_ = np.where(is_)[0]
df = df.drop(is_).reset_index(drop=True)

# Find the index of the video first frame
timestamps = df['Recording timestamp']

# The frame that each row actually corresponds to
first_frame = df.iat[index_first_frame, 17]
df['Video frame'] = np.trunc(df['Video frame th']) + 1 - np.trunc(first_frame)

# Find the index of first frame of Task 1
events = df['Event']
events = events.tolist()
index_t1_s = events.index("Task " + Task + " - Start")
index_t1_e = events.index("Task " + Task + " - End")

# Save gaze location and frames
frames_t1 = list(range(int(df.iat[index_t1_s, 18]), int(df.iat[index_t1_e, 18] + 1)))
frames = df['Video frame']
frames_th = df['Video frame th']
gaze_xs = df['Gaze point X']
gaze_ys = df['Gaze point Y']
gaze_type = df['Eye movement type']
frames = frames.tolist()
# endregion
print(participant, session)
### Record and save
scale_factor = 1
if Task == "1":
    dark_blue = np.array([160, 200, 215])
    light_blue = np.array([50, 110, 40])
    dark_blue = np.array([180, 220, 255])
    light_blue = np.array([30, 90, 0])
else:
    dark_blue = np.array([70, 130, 230])
    light_blue = np.array([25, 70, 50])
# region
args = {"tracker": 'csrt'}
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT.create,
}
tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
initBB = None

df["Hand centre x"] = np.nan
df["Hand centre y"] = np.nan
df["Hand radius"] = np.nan
df["Hand gaze"] = np.nan
df["Board box"] = np.nan
df["Board gaze"] = np.nan

frames_t1_index = list(range(index_t1_s + 1, index_t1_e))

count = -1
fps = None
for c in frames_t1:
    ret, frame = cap.read()
    cap.set(1, c)
    frame = imutils.resize(frame, width=960)
    frame_board = frame.copy()
    (H, W) = frame.shape[:2]
    if ret == True:
        count = count + 1
        if count == 0:  # Skip the first frame
            print("Skip")
            fps = FPS().start()
            continue
        assert isinstance(count, object)
        if initBB is not None:
            (success, box) = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (int(x + 0.5 * w), int(y + 0.5 * h)), radius=8, color=(0, 0, 0),
                           thickness=-1)
                rad = int(sqrt((w / 2) * (w / 2) + (h / 2) * (h / 2)) * scale_factor)
                cv2.circle(frame, (int(x + 0.5 * w), int(y + 0.5 * h)), radius=rad, color=(0, 0, 0),
                           thickness=3)
                try:
                    index_df = frames.index(c)
                    df.at[index_df, "Hand centre x"] = int(x + 0.5 * w)
                    df.at[index_df, "Hand centre y"] = int(y + 0.5 * h)
                    df.at[index_df, "Hand radius"] = rad
                except:
                    pass

                # Check in or out
                try:
                    index_df = frames.index(c)
                    if math.dist([int(x + 0.5 * w), int(y + 0.5 * h)],
                                 [int(gaze_xs[index_df] / 2), int(gaze_ys[index_df]) / 2]) < rad:
                        cv2.circle(frame, (int(x + 0.5 * w), int(y + 0.5 * h)), radius=rad, color=(0, 0, 0),
                                   thickness=3)
                        df.at[index_df, "Hand gaze"] = "In"
                    else:
                        cv2.circle(frame, (int(x + 0.5 * w), int(y + 0.5 * h)), radius=rad, color=(255, 100, 0),
                                   thickness=3)
                        df.at[index_df, "Hand gaze"] = "Out"
                except:
                    try:
                        df.at[index_df, "Hand gaze"] = "No gaze"
                    except:
                        pass
                    pass

        try:
            index_df = frames.index(c)
            if (gaze_type[index_df] == 'Fixation'):  # Green
                cv2.circle(frame, (int(gaze_xs[index_df] / 2), int(gaze_ys[index_df] / 2)), radius=10,
                           color=(0, 255, 0),
                           thickness=-1)
            elif (gaze_type[index_df] == 'Saccade'):  # Blue
                cv2.circle(frame, (int(gaze_xs[index_df] / 2), int(gaze_ys[index_df] / 2)), radius=10,
                           color=(255, 0, 0),
                           thickness=-1)
            else:  # Red
                cv2.circle(frame, (int(gaze_xs[index_df] / 2), int(gaze_ys[index_df] / 2)), radius=10,
                           color=(0, 0, 255),
                           thickness=-1)
        except:
            pass

        if Task == "1" or Task == "2":
            try:
                board_box = Image_processing_func.find_board(frame_board, frame, dark_blue, light_blue)
                index_df = frames.index(c)
                df.at[index_df, "Board box"] = str(board_box)
            except:
                pass

        if Task == "1" or Task == "2":
            try:
                index_df = frames.index(c)
                if cv2.pointPolygonTest(board_box, (int(gaze_xs[index_df] / 2), int(gaze_ys[index_df] / 2)),
                                        False) == 1:
                    # print("In")
                    img_rec = cv2.drawContours(frame.copy(), [board_box], 0, (0, 0, 255), 3)
                    df.at[index_df, "Board gaze"] = "In"
                else:
                    # print("Out")
                    img_rec = cv2.drawContours(frame.copy(), [board_box], 0, (255, 0, 255), 3)
                    df.at[index_df, "Board gaze"] = "Out"
            except:
                img_rec = frame.copy()
                try:
                    df.at[index_df, "Board gaze"] = "No gaze"
                except:
                    pass
                pass
        else:
            img_rec = frame.copy()

        # Info
        fps.update()
        fps.stop()
        info = [
            # ("Tracker", args["tracker"]),
            # ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(img_rec, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.imshow("Frame", img_rec)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") or initBB == None:
            for i in range(10):
                initBB = cv2.selectROI("Frame", frame, fromCenter=True,
                                       showCrosshair=True)
                # After selection
                (xBB, yBB, wBB, hBB) = [int(v) for v in initBB]
                cv2.rectangle(frame_board, (xBB, yBB), (xBB + wBB, yBB + hBB),
                              (0, 255, 0), 2)
                cv2.circle(frame_board, (int(xBB + 0.5 * wBB), int(yBB + 0.5 * hBB)), radius=10,
                           color=(0, 255 - 10 * i, 255 - 10 * i),
                           thickness=-1)
                radBB = int(sqrt((wBB / 2) * (wBB / 2) + (hBB / 2) * (hBB / 2)) * scale_factor)
                cv2.circle(frame_board, (int(xBB + 0.5 * wBB), int(yBB + 0.5 * hBB)), radius=radBB,
                           color=(0, 255 - 10 * i, 255 - 10 * i),
                           thickness=3)

                try:
                    index_df = frames.index(c)
                    df.at[index_df, "Hand centre x"] = int(xBB + 0.5 * wBB)
                    df.at[index_df, "Hand centre y"] = int(yBB + 0.5 * hBB)
                    df.at[index_df, "Hand radius"] = radBB
                except:
                    print("No data in frame", c)

                cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
                cv2.imshow("Frame", frame_board)
                k2 = cv2.waitKey(0)
                if k2 == ord("a"):  # Accept
                    tracker.init(frame, initBB)
                    break
                if k2 == ord("s"):  # Again
                    continue
        elif key == ord("q"):
            break
    else:
        break

# cap.release()
cv2.destroyAllWindows()
# endregion

scale_factor = 1
### Calculate:
# region
events = df['Event']
events = events.tolist()
index_t1_s = events.index("Task " + Task + " - Start")
index_t1_e = events.index("Task " + Task + " - End")

# Save gaze location and frames
frames_t1 = list(range(int(df.iat[index_t1_s, 18]), int(df.iat[index_t1_e, 18] + 1)))

Video_frame_data = []
hand_centre_x_frame_data = []
hand_centre_y_frame_data = []
hand_radius_frame_data = []

board_1_frame_data = []
board_2_frame_data = []
board_3_frame_data = []
board_4_frame_data = []
board_5_frame_data = []
board_6_frame_data = []
board_7_frame_data = []
board_8_frame_data = []

df["Hand centre x - spl"] = np.nan
df["Hand centre y - spl"] = np.nan
df["Hand radius - spl"] = np.nan
df["Hand gaze - spl"] = np.nan

df["Board 1 - spl"] = np.nan
df["Board 2 - spl"] = np.nan
df["Board 3 - spl"] = np.nan
df["Board 4 - spl"] = np.nan
df["Board 5 - spl"] = np.nan
df["Board 6 - spl"] = np.nan
df["Board 7 - spl"] = np.nan
df["Board 8 - spl"] = np.nan
df["Board gaze - spl"] = np.nan

for i in range(index_t1_s + 1, index_t1_e):
    if not (pd.isnull(df.iat[i, 19])) and not (pd.isnull(df.iat[i, 23]) and (Task == '1' or Task == '2')):
        Video_frame_data.append(df.iat[i, 17])

        hand_centre_x_frame_data.append(df.iat[i, 19])
        hand_centre_y_frame_data.append(df.iat[i, 20])
        hand_radius_frame_data.append(df.iat[i, 21])
        if Task == "1" or Task == "2":
            taghi = df.iat[i, 23].replace('\n', ',').replace(']', ',').replace('[', ',').split(',')
            taghi = [s.split() for s in taghi]
            taghi_x = [s[0] for s in taghi if s]
            taghi_y = [s[1] for s in taghi if s]

            board_1_frame_data.append(int(taghi_x[0]))
            board_2_frame_data.append(int(taghi_y[0]))
            board_3_frame_data.append(int(taghi_x[1]))
            board_4_frame_data.append(int(taghi_y[1]))
            board_5_frame_data.append(int(taghi_x[2]))
            board_6_frame_data.append(int(taghi_y[2]))
            board_7_frame_data.append(int(taghi_x[3]))
            board_8_frame_data.append(int(taghi_y[3]))

hand_centre_x_spl = UnivariateSpline(Video_frame_data, hand_centre_x_frame_data)
hand_centre_y_spl = UnivariateSpline(Video_frame_data, hand_centre_y_frame_data)
hand_radius_spl = UnivariateSpline(Video_frame_data, hand_radius_frame_data)

if Task == "1" or Task == "2":
    board_1_spl = UnivariateSpline(Video_frame_data, board_1_frame_data)
    board_2_spl = UnivariateSpline(Video_frame_data, board_2_frame_data)
    board_3_spl = UnivariateSpline(Video_frame_data, board_3_frame_data)
    board_4_spl = UnivariateSpline(Video_frame_data, board_4_frame_data)
    board_5_spl = UnivariateSpline(Video_frame_data, board_5_frame_data)
    board_6_spl = UnivariateSpline(Video_frame_data, board_6_frame_data)
    board_7_spl = UnivariateSpline(Video_frame_data, board_7_frame_data)
    board_8_spl = UnivariateSpline(Video_frame_data, board_8_frame_data)

count_fixate = 0
count_fixate_hand = 0
count_fixate_board = 0
count_fixate_board_not_hand = 0
count_fixate_board_nor_hand = 0
for i in range(index_t1_s + 1, index_t1_e):
    atHand = False
    atBoard = False
    if df.at[i, "Eye movement type"] == 'Fixation':
        count_fixate = count_fixate + 1

    df.at[i, "Hand centre x - spl"] = hand_centre_x_spl(df.iat[i, 17])
    df.at[i, "Hand centre y - spl"] = hand_centre_y_spl(df.iat[i, 17])
    df.at[i, "Hand radius - spl"] = hand_radius_spl(df.iat[i, 17]) * scale_factor

    if pd.isnull(df.at[i, "Gaze point X"]):
        df.at[i, "Hand gaze - spl"] = "No gaze"
    else:
        if math.dist([df.at[i, "Hand centre x - spl"], df.at[i, "Hand centre y - spl"]],
                     [df.at[i, "Gaze point X"] / 2, df.at[i, "Gaze point Y"] / 2]) < df.at[i, "Hand radius - spl"]:
            df.at[i, "Hand gaze - spl"] = "In"
            atHand = True
            if df.at[i, "Eye movement type"] == 'Fixation':
                count_fixate_hand = count_fixate_hand + 1
        else:
            df.at[i, "Hand gaze - spl"] = "Out"

    if Task == "1" or Task == "2":
        df.at[i, "Board 1 - spl"] = board_1_spl(df.iat[i, 17])
        df.at[i, "Board 2 - spl"] = board_2_spl(df.iat[i, 17])
        df.at[i, "Board 3 - spl"] = board_3_spl(df.iat[i, 17])
        df.at[i, "Board 4 - spl"] = board_4_spl(df.iat[i, 17])
        df.at[i, "Board 5 - spl"] = board_5_spl(df.iat[i, 17])
        df.at[i, "Board 6 - spl"] = board_6_spl(df.iat[i, 17])
        df.at[i, "Board 7 - spl"] = board_7_spl(df.iat[i, 17])
        df.at[i, "Board 8 - spl"] = board_8_spl(df.iat[i, 17])
        ccc = [np.array([int(df.at[i, "Board 1 - spl"]), int(df.at[i, "Board 2 - spl"])]),
               np.array([int(df.at[i, "Board 3 - spl"]), int(df.at[i, "Board 4 - spl"])]),
               np.array([int(df.at[i, "Board 5 - spl"]), int(df.at[i, "Board 6 - spl"])]),
               np.array([int(df.at[i, "Board 7 - spl"]), int(df.at[i, "Board 8 - spl"])])]
        board_box_spl = np.array(ccc)

        if pd.isnull(df.at[i, "Gaze point X"]):
            df.at[i, "Board gaze - spl"] = "No gaze"
        else:
            if cv2.pointPolygonTest(board_box_spl,
                                    (int(df.at[i, "Gaze point X"] / 2), int(df.at[i, "Gaze point Y"] / 2)),
                                    False) == 1:
                df.at[i, "Board gaze - spl"] = "In"
                if df.at[i, "Eye movement type"] == 'Fixation':
                    count_fixate_board = count_fixate_board + 1
                    if atHand == False:
                        count_fixate_board_not_hand = count_fixate_board_not_hand + 1
            else:
                df.at[i, "Board gaze - spl"] = "Out"
                if df.at[i, "Eye movement type"] == 'Fixation':
                    if atHand == False:
                        count_fixate_board_nor_hand = count_fixate_board_nor_hand + 1

print("Fixation at hand:", count_fixate_hand / count_fixate * 100)
print("Fixation at board:", count_fixate_board / count_fixate * 100)
print("Fixation at board not hand:", count_fixate_board_not_hand / count_fixate * 100)
print("Fixation at board or hand:", 100 - count_fixate_board_nor_hand / count_fixate * 100)
print(count_fixate_hand / count_fixate * 100, count_fixate_board / count_fixate * 100,
      count_fixate_board_not_hand / count_fixate * 100, 100 - count_fixate_board_nor_hand / count_fixate * 100)
print(participant, session)
# endregion
# region

# Save
df.to_excel('Outputs/' + file[:-5] + " - output.xlsx")
# endregion
print("Saved")

scale_factor = 2
### Play
# region
# video_name = "apdZYHEYdcZj8W4YME6gEA==.mp4" # for vidhi s 1
cap = cv2.VideoCapture('Media/' + video_name)
frames = df['Video frame']
frames = frames.tolist()
img_board = np.full((960, 540, 3), (255, 0, 255), dtype=np.uint8)

events = df['Event']
events = events.tolist()
index_t1_s = events.index("Task " + Task + " - Start")
index_t1_e = events.index("Task " + Task + " - End")

# Save gaze location and frames
frames_t1 = list(range(int(df.iat[index_t1_s, 18]), int(df.iat[index_t1_e, 18] + 1)))

count = -1
fps = None
for c in frames_t1:
    ret, frame = cap.read()
    cap.set(1, c)
    frame = imutils.resize(frame, width=960)
    frame_board = frame.copy()
    (H, W) = frame.shape[:2]
    if ret == True:
        count = count + 1
        if count == 0:  # Skip the first frame
            print("Skip")
            fps = FPS().start()
            continue
        assert isinstance(count, object)

        try:
            index_df = frames.index(c)
            cv2.circle(frame,
                       (int(df.at[index_df, "Hand centre x - spl"]), int(df.at[index_df, "Hand centre y - spl"])),
                       radius=10, color=(0, 0, 0),
                       thickness=-1)
            cv2.circle(frame,
                       (int(df.at[index_df, "Hand centre x - spl"]), int(df.at[index_df, "Hand centre y - spl"])),
                       radius=int(df.at[index_df, "Hand radius - spl"])*scale_factor, color=(0, 0, 0),
                       thickness=3)
        except:
            pass

        try:
            index_df = frames.index(c)
            cv2.circle(frame, (int(df.at[index_df, "Gaze point X"] / 2), int(df.at[index_df, "Gaze point Y"] / 2)),
                       radius=20, color=(0, 255, 0),
                       thickness=-1)
        except:
            pass

        try:
            index_df = frames.index(c)
            ccc = [np.array([int(df.at[index_df, "Board 1 - spl"]), int(df.at[index_df, "Board 2 - spl"])]),
                   np.array([int(df.at[index_df, "Board 3 - spl"]), int(df.at[index_df, "Board 4 - spl"])]),
                   np.array([int(df.at[index_df, "Board 5 - spl"]), int(df.at[index_df, "Board 6 - spl"])]),
                   np.array([int(df.at[index_df, "Board 7 - spl"]), int(df.at[index_df, "Board 8 - spl"])])]
            board_box_spl = np.array(ccc)
            img_board = cv2.drawContours(frame.copy(), [board_box_spl], 0, (255, 0, 255), 3)
        except:
            pass

        # Info
        fps.update()
        fps.stop()
        info = [
            # ("Tracker", args["tracker"]),
            # ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(img_board, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.imshow("Frame", img_board)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    else:
        break
# cap.release()
cv2.destroyAllWindows()
# endregion

# Load
# region
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
import imutils
from imutils.video import FPS


def ellipse_eqn_in(x, y, cnetre_point, maj_ax, min_ax, rot_ang):
    xa = x - cnetre_point[0]
    ya = y - cnetre_point[1]
    term1 = math.pow(xa * math.cos(math.radians(rot_ang)) + ya * math.sin(math.radians(rot_ang)), 2)
    term2 = math.pow(xa * math.sin(math.radians(rot_ang)) - ya * math.cos(math.radians(rot_ang)), 2)
    if (term1 / math.pow(maj_ax, 2)) + (term2 / math.pow(min_ax, 2)) < 1:
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


# endregion
Task = '1'
folder = 'Outputs/Task' + Task + '/'
files = [f for f in listdir(folder) if isfile(join(folder, f))]
file = folder + files[23]
df = pd.read_excel(file, skiprows=[1], index_col=0, skipfooter=1)
print("Loaded")
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
print(video_name)

# Make a list: Outputs
Task = '4'
# region
# region
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
import imutils
from imutils.video import FPS


def ellipse_eqn_in(x, y, cnetre_point, maj_ax, min_ax, rot_ang):
    xa = x - cnetre_point[0]
    ya = y - cnetre_point[1]
    term1 = math.pow(xa * math.cos(math.radians(rot_ang)) + ya * math.sin(math.radians(rot_ang)), 2)
    term2 = math.pow(xa * math.sin(math.radians(rot_ang)) - ya * math.cos(math.radians(rot_ang)), 2)
    if (term1 / math.pow(maj_ax, 2)) + (term2 / math.pow(min_ax, 2)) < 1:
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


# endregion
number = []
par = []
sess = []
file_names = []
at_hands = []
at_boards = []
at_board_not_hands = []
at_board_or_hands = []

folder = 'Outputs/Task' + Task + '/'
files = [f for f in listdir(folder) if isfile(join(folder, f))]
iii = -1
while True:
    iii = iii + 1
    if iii == 141:
        break
    file = folder + files[iii]
    # Examples: 5-problematic
    try:
        df = pd.read_excel(file, skiprows=[1], index_col=0, skipfooter=1)
        print("data is loaded")
        ### Process
        participant = df.iat[1, 5]
        session = str(df.iat[1, 6])[-1]
        print(iii, participant, session)
        number.append(iii)
        par.append(participant)
        sess.append(session)
        file_names.append(files[iii])
        ### Process
        # region

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
        if (participant == "Vidhi" and session == '1'):
            video_name = "apdZYHEYdcZj8W4YME6gEA==.mp4"  # for vidhi s 1
        cap = cv2.VideoCapture('Media/' + video_name)
        # Find video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # The frame that each row should correspond to
        df['Video frame th'] = df['Recording timestamp'] * fps / 1000

        # Keep only eye tracker rows + tasks
        is_ = df['Event'] == "SyncPortOutLow"
        is_ = is_.tolist()
        is_ = np.where(is_)[0]
        df = df.drop(is_).reset_index(drop=True)
        is_ = df['Event'] == "SyncPortOutHigh"
        is_ = is_.tolist()
        is_ = np.where(is_)[0]
        df = df.drop(is_).reset_index(drop=True)

        # Find the index of the video first frame
        timestamps = df['Recording timestamp']

        # The frame that each row actually corresponds to
        first_frame = df.iat[index_first_frame, 17]
        df['Video frame'] = np.trunc(df['Video frame th']) + 1 - np.trunc(first_frame)

        # Find the index of first frame of Task 1
        events = df['Event']
        events = events.tolist()
        index_t1_s = events.index("Task " + Task + " - Start")
        index_t1_e = events.index("Task " + Task + " - End")

        # Save gaze location and frames
        frames_t1 = list(range(int(df.iat[index_t1_s, 18]), int(df.iat[index_t1_e, 18] + 1)))
        frames = df['Video frame']
        frames_th = df['Video frame th']
        gaze_xs = df['Gaze point X']
        gaze_ys = df['Gaze point Y']
        gaze_type = df['Eye movement type']
        frames = frames.tolist()
        # endregion
        print(participant, session)
        ### Calculate:
        # region
        events = df['Event']
        events = events.tolist()
        index_t1_s = events.index("Task " + Task + " - Start")
        index_t1_e = events.index("Task " + Task + " - End")

        # Save gaze location and frames
        frames_t1 = list(range(int(df.iat[index_t1_s, 18]), int(df.iat[index_t1_e, 18] + 1)))

        Video_frame_data = []
        hand_centre_x_frame_data = []
        hand_centre_y_frame_data = []
        hand_radius_frame_data = []

        board_1_frame_data = []
        board_2_frame_data = []
        board_3_frame_data = []
        board_4_frame_data = []
        board_5_frame_data = []
        board_6_frame_data = []
        board_7_frame_data = []
        board_8_frame_data = []

        df["Hand centre x - spl"] = np.nan
        df["Hand centre y - spl"] = np.nan
        df["Hand radius - spl"] = np.nan
        df["Hand gaze - spl"] = np.nan

        df["Board 1 - spl"] = np.nan
        df["Board 2 - spl"] = np.nan
        df["Board 3 - spl"] = np.nan
        df["Board 4 - spl"] = np.nan
        df["Board 5 - spl"] = np.nan
        df["Board 6 - spl"] = np.nan
        df["Board 7 - spl"] = np.nan
        df["Board 8 - spl"] = np.nan
        df["Board gaze - spl"] = np.nan

        for i in range(index_t1_s + 1, index_t1_e):
            if not (pd.isnull(df.iat[i, 19])) and not (pd.isnull(df.iat[i, 23]) and (Task == '1' or Task == '2')):
                Video_frame_data.append(df.iat[i, 17])

                hand_centre_x_frame_data.append(df.iat[i, 19])
                hand_centre_y_frame_data.append(df.iat[i, 20])
                hand_radius_frame_data.append(df.iat[i, 21])
                if Task == "1" or Task == "2":
                    taghi = df.iat[i, 23].replace('\n', ',').replace(']', ',').replace('[', ',').split(',')
                    taghi = [s.split() for s in taghi]
                    taghi_x = [s[0] for s in taghi if s]
                    taghi_y = [s[1] for s in taghi if s]

                    board_1_frame_data.append(int(taghi_x[0]))
                    board_2_frame_data.append(int(taghi_y[0]))
                    board_3_frame_data.append(int(taghi_x[1]))
                    board_4_frame_data.append(int(taghi_y[1]))
                    board_5_frame_data.append(int(taghi_x[2]))
                    board_6_frame_data.append(int(taghi_y[2]))
                    board_7_frame_data.append(int(taghi_x[3]))
                    board_8_frame_data.append(int(taghi_y[3]))

        hand_centre_x_spl = UnivariateSpline(Video_frame_data, hand_centre_x_frame_data)
        hand_centre_y_spl = UnivariateSpline(Video_frame_data, hand_centre_y_frame_data)
        hand_radius_spl = UnivariateSpline(Video_frame_data, hand_radius_frame_data)

        if Task == "1" or Task == "2":
            board_1_spl = UnivariateSpline(Video_frame_data, board_1_frame_data)
            board_2_spl = UnivariateSpline(Video_frame_data, board_2_frame_data)
            board_3_spl = UnivariateSpline(Video_frame_data, board_3_frame_data)
            board_4_spl = UnivariateSpline(Video_frame_data, board_4_frame_data)
            board_5_spl = UnivariateSpline(Video_frame_data, board_5_frame_data)
            board_6_spl = UnivariateSpline(Video_frame_data, board_6_frame_data)
            board_7_spl = UnivariateSpline(Video_frame_data, board_7_frame_data)
            board_8_spl = UnivariateSpline(Video_frame_data, board_8_frame_data)

        count_fixate = 0
        count_fixate_hand = 0
        count_fixate_board = 0
        count_fixate_board_not_hand = 0
        count_fixate_board_nor_hand = 0
        for i in range(index_t1_s + 1, index_t1_e):
            atHand = False
            atBoard = False
            if df.at[i, "Eye movement type"] == 'Fixation':
                count_fixate = count_fixate + 1

            df.at[i, "Hand centre x - spl"] = hand_centre_x_spl(df.iat[i, 17])
            df.at[i, "Hand centre y - spl"] = hand_centre_y_spl(df.iat[i, 17])
            df.at[i, "Hand radius - spl"] = hand_radius_spl(df.iat[i, 17]) * scale_factor

            if pd.isnull(df.at[i, "Gaze point X"]):
                df.at[i, "Hand gaze - spl"] = "No gaze"
            else:
                if math.dist([df.at[i, "Hand centre x - spl"], df.at[i, "Hand centre y - spl"]],
                             [df.at[i, "Gaze point X"] / 2, df.at[i, "Gaze point Y"] / 2]) < df.at[
                    i, "Hand radius - spl"]:
                    df.at[i, "Hand gaze - spl"] = "In"
                    atHand = True
                    if df.at[i, "Eye movement type"] == 'Fixation':
                        count_fixate_hand = count_fixate_hand + 1
                else:
                    df.at[i, "Hand gaze - spl"] = "Out"

            if Task == "1" or Task == "2":
                df.at[i, "Board 1 - spl"] = board_1_spl(df.iat[i, 17])
                df.at[i, "Board 2 - spl"] = board_2_spl(df.iat[i, 17])
                df.at[i, "Board 3 - spl"] = board_3_spl(df.iat[i, 17])
                df.at[i, "Board 4 - spl"] = board_4_spl(df.iat[i, 17])
                df.at[i, "Board 5 - spl"] = board_5_spl(df.iat[i, 17])
                df.at[i, "Board 6 - spl"] = board_6_spl(df.iat[i, 17])
                df.at[i, "Board 7 - spl"] = board_7_spl(df.iat[i, 17])
                df.at[i, "Board 8 - spl"] = board_8_spl(df.iat[i, 17])
                ccc = [np.array([int(df.at[i, "Board 1 - spl"]), int(df.at[i, "Board 2 - spl"])]),
                       np.array([int(df.at[i, "Board 3 - spl"]), int(df.at[i, "Board 4 - spl"])]),
                       np.array([int(df.at[i, "Board 5 - spl"]), int(df.at[i, "Board 6 - spl"])]),
                       np.array([int(df.at[i, "Board 7 - spl"]), int(df.at[i, "Board 8 - spl"])])]
                board_box_spl = np.array(ccc)

                if pd.isnull(df.at[i, "Gaze point X"]):
                    df.at[i, "Board gaze - spl"] = "No gaze"
                else:
                    if cv2.pointPolygonTest(board_box_spl,
                                            (int(df.at[i, "Gaze point X"] / 2), int(df.at[i, "Gaze point Y"] / 2)),
                                            False) == 1:
                        df.at[i, "Board gaze - spl"] = "In"
                        if df.at[i, "Eye movement type"] == 'Fixation':
                            count_fixate_board = count_fixate_board + 1
                            if atHand == False:
                                count_fixate_board_not_hand = count_fixate_board_not_hand + 1
                    else:
                        df.at[i, "Board gaze - spl"] = "Out"
                        if df.at[i, "Eye movement type"] == 'Fixation':
                            if atHand == False:
                                count_fixate_board_nor_hand = count_fixate_board_nor_hand + 1

        # endregion
        print("Fixation at hand:", count_fixate_hand / count_fixate * 100)
        print("Fixation at board:", count_fixate_board / count_fixate * 100)
        print("Fixation at board not hand:", count_fixate_board_not_hand / count_fixate * 100)
        print("Fixation at board or hand:", 100 - count_fixate_board_nor_hand / count_fixate * 100)
        print(count_fixate_hand / count_fixate * 100, count_fixate_board / count_fixate * 100,
              count_fixate_board_not_hand / count_fixate * 100, 100 - count_fixate_board_nor_hand / count_fixate * 100)
        print(participant, session)
        at_hands.append(count_fixate_hand / count_fixate * 100)
        at_boards.append(count_fixate_board / count_fixate * 100)
        at_board_not_hands.append(count_fixate_board_not_hand / count_fixate * 100)
        at_board_or_hands.append(100 - count_fixate_board_nor_hand / count_fixate * 100)
    except:
        pass

number_arr = np.transpose(np.array(number))
participant_arr = np.transpose(np.array(par))
session_arr = np.transpose(np.array(sess))
file_names_arr = np.transpose(np.array(file_names))
at_hands = np.transpose(np.array(at_hands))
at_boards = np.transpose(np.array(at_boards))
at_board_not_hands = np.transpose(np.array(at_board_not_hands))
at_board_or_hands = np.transpose(np.array(at_board_or_hands))
final = np.transpose(np.vstack((number_arr, participant_arr, session_arr, file_names_arr, at_hands, at_boards,
                                at_board_not_hands, at_board_or_hands)))
df = pd.DataFrame(final)
filepath = 'Task' + Task + '_outputs_12.xlsx'
df.to_excel(filepath, index=False)
# endregion

# Make a list: Raws
# region
Task = '3'
# region
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
import imutils
from imutils.video import FPS


def ellipse_eqn_in(x, y, cnetre_point, maj_ax, min_ax, rot_ang):
    xa = x - cnetre_point[0]
    ya = y - cnetre_point[1]
    term1 = math.pow(xa * math.cos(math.radians(rot_ang)) + ya * math.sin(math.radians(rot_ang)), 2)
    term2 = math.pow(xa * math.sin(math.radians(rot_ang)) - ya * math.cos(math.radians(rot_ang)), 2)
    if (term1 / math.pow(maj_ax, 2)) + (term2 / math.pow(min_ax, 2)) < 1:
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


# endregion
number = []
par = []
sess = []
file_names = []

folder = 'Task' + Task + '/'
files = [f for f in listdir(folder) if isfile(join(folder, f))]
iii = -1
while True:
    iii = iii + 1
    if iii == 141:
        break
    file = folder + files[iii]
    # Examples: 5-problematic
    df = pd.read_excel(file, skiprows=[1], skipfooter=1)
    print("data is loaded")
    try:
        ### Process
        # region
        participant = df.iat[1, 5]
        session = str(df.iat[1, 6])[-1]
        print(iii, participant, session)
        number.append(iii)
        par.append(participant)
        sess.append(session)
        file_names.append(files[iii])
    except:
        pass
number_arr = np.transpose(np.array(number))
participant_arr = np.transpose(np.array(par))
session_arr = np.transpose(np.array(sess))
final = np.transpose(np.vstack((number_arr, participant_arr, session_arr)))
df = pd.DataFrame(final)
filepath = 'Task' + Task + '_raw_key.xlsx'
df.to_excel(filepath, index=False)
# endregion

# Make a list: Efficiencies
Task = '4'
# region
# region
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
import imutils
from imutils.video import FPS


def ellipse_eqn_in(x, y, cnetre_point, maj_ax, min_ax, rot_ang):
    xa = x - cnetre_point[0]
    ya = y - cnetre_point[1]
    term1 = math.pow(xa * math.cos(math.radians(rot_ang)) + ya * math.sin(math.radians(rot_ang)), 2)
    term2 = math.pow(xa * math.sin(math.radians(rot_ang)) - ya * math.cos(math.radians(rot_ang)), 2)
    if (term1 / math.pow(maj_ax, 2)) + (term2 / math.pow(min_ax, 2)) < 1:
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


# endregion
number = []
par = []
sess = []
file_names = []
eff = []

folder = 'Task' + Task + '/'
files = [f for f in listdir(folder) if isfile(join(folder, f))]
iii = -1
while True:
    iii = iii + 1
    if iii == 141:
        break
    file = folder + files[iii]
    # Examples: 5-problematic
    df = pd.read_excel(file, skiprows=[1], skipfooter=1)
    print("data is loaded")
    c_eff = 0
    c_tot = 0
    try:
        ### Process
        # region
        participant = df.iat[1, 5]
        session = str(df.iat[1, 6])[-1]

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
        if (participant == "Vidhi" and session == '1'):
            video_name = "apdZYHEYdcZj8W4YME6gEA==.mp4"  # for vidhi s 1
        cap = cv2.VideoCapture('Media/' + video_name)
        # Find video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # The frame that each row should correspond to
        df['Video frame th'] = df['Recording timestamp'] * fps / 1000

        # Keep only eye tracker rows + tasks
        is_ = df['Event'] == "SyncPortOutLow"
        is_ = is_.tolist()
        is_ = np.where(is_)[0]
        df = df.drop(is_).reset_index(drop=True)
        is_ = df['Event'] == "SyncPortOutHigh"
        is_ = is_.tolist()
        is_ = np.where(is_)[0]
        df = df.drop(is_).reset_index(drop=True)

        # Find the index of the video first frame
        timestamps = df['Recording timestamp']

        # The frame that each row actually corresponds to
        first_frame = df.iat[index_first_frame, 17]
        df['Video frame'] = np.trunc(df['Video frame th']) + 1 - np.trunc(first_frame)

        # Find the index of first frame of Task 1
        events = df['Event']
        events = events.tolist()
        index_t1_s = events.index("Task " + Task + " - Start")
        index_t1_e = events.index("Task " + Task + " - End")

        # Save gaze location and frames
        frames_t1 = list(range(int(df.iat[index_t1_s, 18]), int(df.iat[index_t1_e, 18] + 1)))
        frames = df['Video frame']
        frames_th = df['Video frame th']
        gaze_xs = df['Gaze point X']
        gaze_ys = df['Gaze point Y']
        gaze_type = df['Eye movement type']
        frames = frames.tolist()
        # endregion
    except:
        pass
    print(participant, session)
    frames_t1_index = list(range(index_t1_s + 1, index_t1_e))

    count = -1
    fps = None
    for c in frames_t1:
        try:
            c_tot = c_tot + 1
            index_df = frames.index(c)
            if not math.isnan(gaze_xs[index_df]):
                c_eff = c_eff + 1
        except:
            c_eff = c_eff + 1
    number.append(iii)
    par.append(participant)
    sess.append(session)
    file_names.append(files[iii])
    try:
        eff.append(c_eff / c_tot * 100)
    except:
        eff.append("NA")

number_arr = np.transpose(np.array(number))
participant_arr = np.transpose(np.array(par))
session_arr = np.transpose(np.array(sess))
eff = np.transpose(np.array(eff))
final = np.transpose(np.vstack((number_arr, participant_arr, session_arr, eff)))
df = pd.DataFrame(final)
filepath = 'Task' + Task + '_Efficiency.xlsx'
df.to_excel(filepath, index=False)
# endregion
