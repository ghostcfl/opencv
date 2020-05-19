import cv2, sys
import numpy as np
from matplotlib import pyplot as plt
import time, os
import glob


def read_jpg(path):
    path_to_read = os.path.join(os.path.dirname(__file__), '1', path + ".jpg")
    im = cv2.imread(path_to_read)
    return im


def split_jpg(im):
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, in_inv = cv2.threshold(im_gray, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = 1 / 16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    im_blur = cv2.filter2D(in_inv, -1, kernel)
    ret, im_res = cv2.threshold(im_blur, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(im_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # img = cv2.drawContours(im, contours, -1, (255, 0, 0), 1)
    result = []
    w_total = []
    word_num = 6
    temp = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        temp.append((x, y, w, h))
        w_total.append(w)
    temp.sort()
    avg = sum(w_total) / word_num + 5
    for _ in temp:
        x, y, w, h = _[0], _[1], _[2], _[3]
    # for contour in contours:
    #     x, y, w, h = cv2.boundingRect(contour)

        if (avg + 5) < w < (avg * 2 + 3):
            box_left = np.int0([[x, y], [x + w / 2, y], [x + w / 2, y + h], [x, y + h]])
            box_right = np.int0([[x + w / 2, y], [x + w, y], [x + w, y + h], [x + w / 2, y + h]])
            result.append(box_left)
            result.append(box_right)
        elif (avg * 2 + 3) < w < (avg * 3):
            box_left = np.int0([[x, y], [x + w / 3, y], [x + w / 3, y + h], [x, y + h]])
            box_mid = np.int0([[x + w / 3, y], [x + w * 2 / 3, y], [x + w * 2 / 3, y + h], [x + w / 3, y + h]])
            box_right = np.int0([[x + w * 2 / 3, y], [x + w, y], [x + w, y + h], [x + w * 2 / 3, y + h]])
            result.append(box_left)
            result.append(box_mid)
            result.append(box_right)
        else:
            box = np.int0([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            result.append(box)
    return result, im_res


def save_split(boxes):
    for box in boxes:
        a = cv2.drawContours(im, [box], 0, (0, 0, 255), 1)
        roi = im_res[box[0][1]:box[3][1], box[0][0]:box[1][0]]
        roistd = cv2.resize(roi, (30, 30))  # 将字符图片统一调整为30x30的图片大小
        timestamp = int(time.time() * 1e6)  # 为防止文件重名，使用时间戳命名文件名
        filename = "{}.jpg".format(timestamp)
        filepath = os.path.join("char", filename)
        cv2.imwrite(filepath, roistd)
    # cv2.imwrite(filename, a)


def label_jpg():
    files = os.listdir("char")
    for filename in files:
        filename_ts = filename.split(".")[0]
        patt = "label/{}_*".format(filename_ts)
        saved_num = len(glob.glob(patt))
        if saved_num == 1:
            print("{}done".format(patt))
            continue
        filepath = os.path.join("char", filename)
        im = cv2.imread(filepath)
        cv2.imshow("image", im)
        key = cv2.waitKey(0)
        if key == 27:
            sys.exit()
        if key == 13:
            continue
        char = chr(key)
        filename_ts = filename.split(".")[0]
        outfile = "{}_{}.jpg".format(filename_ts, char)
        outpath = os.path.join("label", outfile)
        cv2.imwrite(outpath, im)


def train():
    filenames = os.listdir("label")
    samples = np.empty((0, 900))
    labels = []
    for filename in filenames:
        filepath = os.path.join("label", filename)
        label = filename.split(".")[0].split("_")[-1]
        labels.append(label)
        im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        sample = im.reshape((1, 900)).astype(np.float32)
        samples = np.append(samples, sample, 0)
        samples = samples.astype(np.float32)
        unique_label = list(set(labels))
        unique_ids = list(range(len(unique_label)))
        label_id_map = dict(zip(unique_label, unique_ids))
        id_label_map = dict(zip(unique_ids, unique_label))
        label_ids = list(map(lambda x: label_id_map[x], labels))
        label_ids = np.array(label_ids).reshape((-1, 1)).astype(np.float32)

        # print(label_ids.shape)
        # print(samples.shape)
    knn = cv2.ml.KNearest_create()
    knn.train(samples, cv2.ml.ROW_SAMPLE, label_ids)
    boxes, im_res = split_jpg(read_jpg(str(21)))
    for box in boxes:
        roi = im_res[box[0][1]:box[3][1], box[0][0]:box[1][0]]
        roistd = cv2.resize(roi, (30, 30))
        sample = roistd.reshape((1, 900)).astype(np.float32)
        ret, results, neighbours, distances = knn.findNearest(sample, k=1)
        label_id = int(results[0, 0])
        label = id_label_map[label_id]
        print(label, end="")


if __name__ == '__main__':
    # for i in range(1,21):
    #     split_jpg(read_jpg(str(i)))
    # split_jpg(read_jpg(str(1)))
    # label_jpg()
    train()
