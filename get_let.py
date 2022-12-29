import numpy as np
from typing import List, Tuple, Any
import cv2
from keras import models
import numpy as np
from multiprocessing import *
from threading import Thread
import time


model = models.load_model('NN5.h5')
emnist_labels = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]


def extract(path: str, out_size: int = 28) -> List[Tuple]:
    """Проходит по изображению, выделяя контуром все объекты"""
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    cv2.waitKey()
    thresh = 100
    ret, thresh = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((1, 1), np.uint8), iterations=2)
    cv2.imshow('erode', img_erode)
    cv2.waitKey()
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    output = img.copy()
    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                y_pos = size_max // 2 - h // 2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif h > w:
                x_pos = size_max // 2 - w // 2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop
            letters.append([x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA), x + y * 100, h, w])
            # cv2.imshow('-', letters[-1][2])
            # cv2.waitKey()
    letters.sort(key=lambda x: x[-3], reverse=False)
    return letters


def emnist_predict_img(i, model=model):
    """Подготовка и само распознавание"""
    global letters
    img = letters[i][2]
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr / 255.0
    img_arr[0] = np.rot90(img_arr[0], 3)
    img_arr[0] = np.fliplr(img_arr[0])
    img_arr = img_arr.reshape((1, 28, 28, 1))
    predict = model.predict([img_arr])
    result = np.argmax(predict, axis=1)
    letters[i].append(chr(emnist_labels[result[0]]))
    return chr(emnist_labels[result[0]])


def parall(path: str):
    """Запускает 3 потока для распознавания букв с изображения"""
    global letters
    letters = extract(path=path)
    start = time.perf_counter()
    for i in range(0, len(letters), 3):
        get1 = Thread(target=emnist_predict_img, args=(i,))
        
        if i + 2 < len(letters):
            get2 = Thread(target=emnist_predict_img, args=(i + 1,))
            get3 = Thread(target=emnist_predict_img, args=(i + 2,))
            get1.start()
            get2.start()
            get3.start()
            get1.join()
            get2.join()
            get3.join()
        elif i + 1 < len(letters):
            get2 = Thread(target=emnist_predict_img, args=(i + 1,))
            get1.start()
            get2.start()
            get1.join()
            get2.join()
        else:
            get1.start()
            get1.join()
        
    finish = time.perf_counter()
    s = ''
    for i in range(len(letters)):
        if letters[i][-2] * letters[i][-3] < letters[0][-2] * letters[0][-3] / 8:
            continue
        dn = letters[i + 1][0] - letters[i][0] - letters[i][1] if i < len(letters) - 1 else 0
        s += letters[i][-1]
        if dn > letters[i][1] // 2 :
            s += ' '
        if len(letters) > i + 1 and letters[i][0] > letters[i + 1][0]:
            s += '\n'
    print(finish - start)
    return s


if __name__ == '__main__':
    print(parall('tests/test_4_rep.png'))