"""
mapping = {
    0: Not Fractured,
    1: Fractured
    }
"""

import numpy as np
import cv2 as cv
import pandas as pd
import os


def get_data(path_dir):
    img_path =[]
    label =[]
    for folder in os.listdir(path_dir):
        for img in os.listdir(os.path.join(path_dir, folder)):
            img_path.append(img)
            label.append(folder)
    data = {"img_path":img_path, "label":label}
    return pd.DataFrame(data).sample(frac=1).reset_index(drop=True)
data = get_data('../BoneFractureDataset/training')
print(data.head(5))

images = []
labels = []
n = 0
for index , row in data.iterrows():

    img_path = row["img_path"]
    label = row["label"]
    img = cv.imread(r'BoneFractureDataset/training/{}/{}'.format(label,img_path))
    max_pixel = 255.0
    img = cv.cvtColor(cv.resize(img,(299,299)), cv.COLOR_BGR2RGB)
    img_norm = img.astype("float32")/max_pixel
    print(f"processing image {n} with shape: {img_norm.shape}")
    images.append(np.array(img_norm))
    labels.append(int(label))
    n += 1

    if n == 3000:
        break

np.save("../Train299X.npy", np.array(images))
np.save("../Train299Y.npy", np.array(labels))
