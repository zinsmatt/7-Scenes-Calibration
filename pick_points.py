import numpy as np
import os
import cv2
import matplotlib.pyplot as plt



folder = "/media/mzins/DATA1/7-Scenes/chess/seq-01"

idx = 0

rgb = cv2.imread(os.path.join(folder, "frame-%06d.color.png" % idx))
depth = cv2.imread(os.path.join(folder, "frame-%06d.depth.png" % idx), cv2.IMREAD_UNCHANGED)
depth[depth == 65535] = 5000


fig = plt.figure()


# plt.imshow(depth)
plt.imshow(rgb[:, :, ::-1])

coords = []

def onclick(event):
    global ix, iy
    global coords
    ix, iy = event.xdata, event.ydata
    coords.append((ix, iy))
    print(len(coords), " ===> ", ix, iy)

    # np.savetxt("points_depth.txt", np.vstack(coords))
    np.savetxt("points_rgb.txt", np.vstack(coords))
    return coords


cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
