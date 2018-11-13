import sys
import cv2
import numpy as np


# we werken aan de hand van otsu's binarization
def autothresh(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    hist = cv2.calcHist([blur], [0], None, [256], [0, 256])
    hist_norm = hist.ravel()/hist.max()
    Q = hist_norm.cumsum()

    bins = np.arange(256)

    fn_min = np.inf
    thresh = -1

    for i in range(1, 256):
        p1, p2 = np.hsplit(hist_norm, [i])  #probabilities
        q1, q2 = Q[i], Q[255] - Q[i]    #cum sum of classes
        q1 = float(q1)
        q2 = float(q2)
        b1, b2 = np.hsplit(bins, [i])   #weights

        # finding means and variances
        m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
        v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2

        # calculates the minimization function
        fn = v1 * q1 + v2 * q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    ret, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu


image = cv2.imread(str(sys.argv[1]), 0)
result = autothresh(image)
cv2.imshow('original', image)
cv2.imshow('result', result)
while (True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# kan intressant zijn om otsu niet voor volledige foto maar voor segmenten te berekenen, dus bv otsu met regio rond pixel, zo kan voor orc_scanner de schaduwregio onder de hand gefixt worden