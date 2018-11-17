import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt


# this method was based on doing equalization on the original image, and then using reverse equalization based on the cartoon image,
# this way the original image would end up with a cartoon-like histogram, this however did not work because of bad
# equalization
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

def ecdf(x):
    """convenience function for computing the empirical CDF"""
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
    return vals, ecdf

def histogram_match_single_color2(source, template):
    # bron = source
    # cartoon = template
    numbertype = source.dtype
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    # return_value = interp_t_values[bin_idx].reshape(oldshape).astype(numbertype)
    # plt.plot(np.ndarray.flatten((cv2.calcHist([cartoon], [0], None, [256], [0, 256])).astype(int))/bron.size, label='org')
    # plt.plot(np.ndarray.flatten((cv2.calcHist([return_value], [0], None, [256], [0, 256])).astype(int))/bron.size, label='eq_org')
    # plt.legend()
    # plt.show()
    return interp_t_values[bin_idx].reshape(oldshape).astype(numbertype)




def histogram_match(orgimage, cartimage):
    colorsplit_cartoon = cv2.split(cartimage)
    colorsplit_orgimage = cv2.split(orgimage)
    colorsplit_final_image = []
    for i in range(0, 3):
        colorsplit_final_image.append(histogram_match_single_color2(colorsplit_orgimage[i], colorsplit_cartoon[i]))
    image = cv2.merge(colorsplit_final_image)
    cv2.imshow('color', image)
    cv2.imshow('cartoonimmage', cartimage)
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
