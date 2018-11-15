import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt


# we werken aan de hand van otsu's binarization
def histogram_match_single_color(orgimage, cartimage):
    eq_orgimage = cv2.equalizeHist(orgimage)       #equalizes the original image to a uniform histogram image
    hist = np.ndarray.flatten((cv2.calcHist([cartimage], [0], None, [256], [0, 256])).astype(int))
    # plt.plot(hist)
    # plt.show()

    amount_pixels = cartimage.size
    Pm_array = np.divide(hist, amount_pixels)
    Cm = np.cumsum(Pm_array)
    L = (hist.size-1) * Cm
    L_round = np.round(L).astype(int)
    pixel_map = {}
    for i in range(0, hist.size):
        a = L_round[i]
        if pixel_map.get(a) is not None:
            pixel_map[a] = pixel_map.get(a) + hist[i]
        else:
            pixel_map[a] = hist[i]

    # cv2.imshow("cartimage", cartimage)
    # equalized_cartimage = np.zeros(cartimage.shape, dtype='uint8')
    # for x in range(0, cartimage.shape[0]):
    #     for y in range(0, cartimage.shape[1]):
    #         a = L_round[cartimage[x][y]]
    #         equalized_cartimage[x][y] = a
    #
    # cv2.imshow('equalized_manual', equalized_cartimage)
    # cv2.imshow('equalized_cv2', cv2.equalizeHist(cartimage))
    # now that we can calculate the equalized image, we need to do the reverse on the original image

    final_image = np.zeros(orgimage.shape, dtype=orgimage.dtype)
    list_of_keys = np.asarray(list(pixel_map.keys()))
    for x in range(0, orgimage.shape[0]):
        for y in range(0, orgimage.shape[1]):
            if pixel_map.get(eq_orgimage[x][y]) is not None:
                final_image[x][y] = pixel_map.get(eq_orgimage[x][y])
            else:
                idx = np.argmin((np.abs(list_of_keys - eq_orgimage[x][y])))     #we proberen de best passende key te vinden
                # in de table voor het omzetten van equalized-histogram naar cartoon-histogram
                final_image[x][y] = pixel_map.get(list_of_keys[idx])
    cv2.imshow('final image', final_image)
    cv2.waitKey(1)
    return final_image

def histogram_match(orgimage, cartimage):
    colorsplit_cartoon = cv2.split(cartimage)
    colorsplit_orgimage = cv2.split(orgimage)

    for i in range(0, 3):
        colorsplit_orgimage[i] = histogram_match_single_color(colorsplit_orgimage[i], colorsplit_cartoon[i])
    image = cv2.merge(colorsplit_orgimage)
    cv2.imshow('color', image)
    cv2.waitKey(1)
    return image
# now that we calculated the equalized histogram for the orgimage, we need to find the transform function from the cartoon to equalized image.
# then we apply this filter in reverse on the equalized original images(split by color) and find the histogram matched final images.
# finally the 3 seperate color images need to be combined


orgimage = cv2.imread(str(sys.argv[1]))
cartimage = cv2.imread(str(sys.argv[2]))
step1_image = histogram_match(orgimage, cartimage)

cv2.imshow('original', orgimage)
cv2.imshow('cartoon image', cartimage)
cv2.imshow('result', step1_image)
while (True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
