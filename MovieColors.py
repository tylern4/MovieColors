from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import trange, tqdm

try:
    import boost_histogram as bh
    histColors = bh.numpy.histogram
except ImportError:
    print("Using Numpy hists")
    histColors = np.histogram


def GetMax(img):
    img = np.round(img, 3)
    img = img.reshape(img.shape[0]*img.shape[1], img.shape[2])

    c = Counter(map(tuple, img))
    maximum = c.most_common()[0][0]
    rgb = [x for x in maximum]
    rgb.append(1)

    return np.array(rgb).T


def count_frames(video):
    total = 0
    # loop over the frames of the video
    while True:
        # grab the current frame
        (grabbed, frame) = video.read()

        # check to see if we have reached the end of the
        # video
        if not grabbed:
            break
        # increment the total number of frames read
        total += 1
    # return the total number of frames in the video file
    video.set(1, 0)
    return total


def half_size(frame):
    return cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))


def get_frames(video):
    total = count_frames(video)
    colors = np.zeros((192, total, 4))
    video.set(1, 0)
    frames = []
    for i in trange(total):
        (grabbed, frame) = video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)/256
        frame = half_size(frame)
        colors[:, i] = GetMax(frame)

    return colors, total


cap = cv2.VideoCapture(
    "/Users/tylern/Downloads/Do You Love Me-fn3KWM1kuAw.mkv")

# cap = cv2.VideoCapture(
#     "/Users/tylern/Desktop/Wire-kGj_HkKhhSE.mkv")


frames, total = get_frames(cap)

# colors = np.zeros((192, len(frames), 4))
# for i, f in tqdm(enumerate(frames)):
#     colors[:, i] = GetMax(f)

fig, ax = plt.subplots(figsize=[192, total//100])
plt.imshow(frames)
plt.axis("off")
plt.savefig("pic.png", bbox_inches='tight', pad_inches=0)
