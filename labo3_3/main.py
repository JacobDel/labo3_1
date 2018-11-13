import sys
import cv2
import numpy as np


# we werken aan de hand van otsu's binarization
def histogram_match_single_color(orgimage, cartimage):
    eq_orgimage = cv2.equalizeHist(orgimage)       #equalizes the original image to a uniform histogram image

    hist, bins = np.histogram([cartimage], bins=256)            #ik probeer een histogram in te lezen maar iets klopt nog niet :(
    cdf = hist.cumsum()

    # code hieronder komt van interner
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')                #this contains the transformation function from cartoon image to equalized image

    hist, bins = np.histogram([orgimage])
    numberlist = []

    for (x,y), value in np.ndenumerate(orgimage):
        if orgimage[x][y] not in numberlist:
            numberlist.append(orgimage[x][y])

    hist = cv2.calcHist(orgimage, channels=3, histSize=[256, 256, 256])
    mp = []
    # for r in range(0, 255):

    return None


def color_split(image):
    colorsplit_images = []
    for i in range(0, 3):
        colorsplit_images.append(np.zeros(shape=(image.shape[0], image.shape[1]), dtype='uint8'))
    for (x, y, z), value in np.ndenumerate(image):
        for i in range(0, 3):
            a = image[x][y][i]
            colorsplit_images[i][x - 1][y - 1] = a
    return colorsplit_images


def histogram_match(orgimage, cartimage):
    colorsplit_cartoon = color_split(cartimage)
    colorsplit_orgimage = color_split(orgimage)

    for i in range(0, 3):
        colorsplit_orgimage[i] = histogram_match_single_color(colorsplit_orgimage[i], colorsplit_cartoon[i])
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
