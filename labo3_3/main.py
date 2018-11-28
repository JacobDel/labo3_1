import sys
import cv2
import numpy as np
import os



def histogram_match_single_color(orgimage, cartimage):
    eq_orgimage = cv2.equalizeHist(orgimage)       #equalizes the original image to a uniform histogram image
    hist = np.ndarray.flatten((cv2.calcHist([cartimage], [0], None, [256], [0, 256])).astype(int))

    amount_pixels = cartimage.size
    Pm_array = np.divide(hist, amount_pixels)
    Cm = np.cumsum(Pm_array)
    L = (hist.size-1) * Cm
    L_round = np.round(L).astype(cartimage.dtype)

    pixel_map = {}
    for pixel_value in np.unique(L_round):
        indexes = np.where(L_round == pixel_value)[0]
        pixel_map[pixel_value] = np.median(indexes).astype(L_round.dtype)

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
    return final_image

# def histogram_match_single_color2(source, template):
#     # bron = source
#     # cartoon = template
#     numbertype = source.dtype
#     oldshape = source.shape
#     source = source.ravel()
#     template = template.ravel()
#
#     # get the set of unique pixel values and their corresponding indices and
#     # counts
#     s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
#                                             return_counts=True)
#     t_values, t_counts = np.unique(template, return_counts=True)
#
#     # take the cumsum of the counts and normalize by the number of pixels to
#     # get the empirical cumulative distribution functions for the source and
#     # template images (maps pixel value --> quantile)
#     s_quantiles = np.cumsum(s_counts).astype(np.float64)
#     s_quantiles /= s_quantiles[-1]
#     t_quantiles = np.cumsum(t_counts).astype(np.float64)
#     t_quantiles /= t_quantiles[-1]
#
#     # interpolate linearly to find the pixel values in the template image
#     # that correspond most closely to the quantiles in the source image
#     interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
#     return interp_t_values[bin_idx].reshape(oldshape).astype(numbertype)


def histogram_match(orgimage,cartimage):
    cartimage_hsv = cv2.cvtColor(cartimage, cv2.COLOR_RGB2HSV)
    orgimage_hsv = cv2.cvtColor(orgimage, cv2.COLOR_RGB2HSV)
    colorsplit_cartoon_hsv = cv2.split(cartimage_hsv)
    colorsplit_orgimage_hsv = cv2.split(orgimage_hsv)
    split_final_image = []
    for i in range(0, 3):
        split_final_image.append(histogram_match_single_color(colorsplit_orgimage_hsv[i], colorsplit_cartoon_hsv[i]))
    # split_final_image[2] = colorsplit_orgimage_hsv[2]
    image = cv2.merge(split_final_image)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.imshow("output", image)
    cv2.waitKey(1)
    return image


orgimage = cv2.imread(str(sys.argv[2]))
cartimage = cv2.imread(str(sys.argv[1]))
step1_image = histogram_match(orgimage, cartimage)
# get the edges of the image
edges = cv2.Canny(step1_image, 50, 500)
# by multiplying both images we get black edges (because the value of black is 0), but white is 255 so:
multiplyImage = np.where(edges > 100, 0, 1) # or we could use np.clip
stacked_img = np.stack((multiplyImage,)*3, axis=-1)
result2 = np.multiply(step1_image, stacked_img)
np.clip(result2, 0, 256)
result2 = result2.astype('uint8')

# bilateral filter
result3 = cv2.bilateralFilter(result2,9,75,75)


cv2.imwrite(os.path.join(os.getcwd(), "result.png"), result3)

cv2.imshow('original', orgimage)
cv2.imshow('cartoon image', cartimage)
cv2.imshow('result1', cv2.resize(step1_image, (step1_image.shape[1], step1_image.shape[0])))
cv2.imshow('edges result1', cv2.resize(edges, (edges.shape[1], edges.shape[0])))
cv2.imshow('result2', cv2.resize(result2, (result2.shape[1], result2.shape[0])))
cv2.imshow('result3', cv2.resize(result3, (result3.shape[1], result3.shape[0])))
while (True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
