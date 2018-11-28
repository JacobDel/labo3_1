import sys
import cv2
import os


image = cv2.imread(str(sys.argv[1]), 0)
cv2.imshow('original', image)

binary_threshold_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite(os.path.join(os.getcwd(), "binary_threshold_image.png"), binary_threshold_image)
cv2.imshow('binary', binary_threshold_image)
otsu_threshold_image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)[1]
cv2.imwrite(os.path.join(os.getcwd(), "otsu_threshold_image.png"), otsu_threshold_image)
cv2.imshow('otsu', otsu_threshold_image)
while (True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# kan intressant zijn om otsu niet voor volledige foto maar voor segmenten te berekenen, dus bv otsu met regio rond pixel, zo kan voor orc_scanner de schaduwregio onder de hand gefixt worden