from __future__ import print_function
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import collections
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD



def resize_region(region):
    resized = cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)
    return resized

def scale_to_range(image):
    return image / 255

def matrix_to_vector(image):
    return image.flatten()

def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
    return ready_for_ann

def convert_output(alphabet):
    #return np.eye(len(outputs))
    outputs = []
    for i in range(len(alphabet)):
        one = np.zeros(len(alphabet))
        one[i] = 1
        outputs.append(one)
    return np.array(outputs)

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

def display_result(outputs, alphabet, k_means):
    w_space_group = max(enumerate(k_means.cluster_centers_), key = lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[1:,:]):
        if (k_means.labels_[idx] == w_space_group):
            result += ' '
        result += alphabet[winner(output)]
    return result



def select_roi(image_orig, image_bin):
    '''
    Funkcija iscrtava pravougaonike na originalnoj slici, pronalazi sortiran niz regiona sa slike,
    i dodatno treba da sačuva rastojanja između susednih regiona.
    '''
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []
    for contour in contours: 
        x,y,w,h = cv2.boundingRect(contour)
        region = image_bin[y:y+h+1,x:x+w+1]
        regions_array.append([resize_region(region), (x,y,w,h)])
        cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)

    regions_array = sorted(regions_array, key=lambda item: item[1][0])

    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    # Izdvojiti sortirane parametre opisujućih pravougaonika
    # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
    for index in range(0, len(sorted_rectangles)-1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index+1]
        distance = next_rect[0] - (current[0]+current[2]) #X_next - (X_current + W_current)
        region_distances.append(distance)

    return image_orig, sorted_regions, region_distances




##################################################################
#######################  MOJE FUNKCIJE   #########################
##################################################################


def return_trained_letters(path):
    img = cv2.imread(path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    img_gs = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2GRAY)
    img_t = 1 - img_gs

    ret, img_bin = cv2.threshold(img_t, 254, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    image_orig, letters, region_distances = my_select_roi(img, img_bin)
    #plt.imshow(image_orig)
    #plt.show()

    return letters


def my_select_roi(img, img_bin):
    reversed_bin = 255 - img_bin
    img2, contours, hierarchy = cv2.findContours(reversed_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
    #plt.imshow(img, 'gray')
    #plt.show()

    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if area > 100:
            region = img_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])

    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]

    try:
        final_rectangles = letters_with_umlaut(sorted_rectangles)
    except Exception as e:
        final_rectangles = sorted_rectangles

    new_sorted_regions = []
    img_original_copy = img.copy()

    for rectangle in final_rectangles:
        region = img_bin[rectangle[1]:rectangle[1] + rectangle[3] + 2, rectangle[0]:rectangle[0] + rectangle[2] + 2]    # region = img_bin[y:y + h + 1, x:x + w + 1]
        new_sorted_regions.append(resize_region(region))
        cv2.rectangle(img, (rectangle[0], rectangle[1]),
                      (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]), (0, 255, 0), 2)

    #https://stackoverflow.com/questions/23720875/how-to-draw-a-rectangle-around-a-region-of-interest-in-python
    for rectangle in sorted_rectangles:
        cv2.rectangle(img_original_copy, (rectangle[0], rectangle[1]),
                      (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]), (0, 255, 0), 2)

    region_distances = []

    for i in range(0, len(final_rectangles) - 1):
        current = final_rectangles[i]
        next_rect = final_rectangles[i + 1]
        distance = next_rect[0] - (current[0] + current[2])
        region_distances.append(distance)

    #plt.imshow(img)
    #plt.show()
    print("Number of recognized rectangles:", len(final_rectangles))

    return img, new_sorted_regions, region_distances



#https://stackoverflow.com/questions/55376338/how-to-join-nearby-bounding-boxes-in-opencv-python
#nije direktno kopiran kod, vec je prilagodjen potrebama
def letters_with_umlaut(sorted_rectangles):
    final_rectangles = sorted_rectangles.copy()
    rectangles_umlauts = []
    for i in range(0, len(sorted_rectangles) - 1):
        this_rect = sorted_rectangles[i]
        next_rect = sorted_rectangles[i + 1]
        if (this_rect[0] + this_rect[2] + 6 > next_rect[0] + next_rect[2])  and (this_rect[0] < next_rect[0]):
            rectangles_umlauts.append(next_rect)
            new_rect = (this_rect[0], next_rect[1], this_rect[2], this_rect[3] + next_rect[3] + 5)
            final_rectangles[i] = new_rect

    if len(rectangles_umlauts) > 0:
        for rect in rectangles_umlauts:
            final_rectangles.remove(rect)

    return final_rectangles


#https://developpaper.com/python-opencv-implementation-of-rotating-text-correction/
def rotation(image_path):
    img = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    img_gs = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2GRAY)
    img_t = 1 - img_gs
    thresh = cv2.adaptiveThreshold(img_t, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 39, 3)
    img_bin = 255 - thresh
    ret, img_bin = cv2.threshold(img_bin, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_t = cv2.bitwise_not(gray)

    # #ret, img_bin = cv2.threshold(img_t, 250, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)


    imgasdad, contours, hierarchy = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for contour in contours: 
        angle = cv2.minAreaRect(contour)[-1]
        if angle < -45:
            angle = -(90+ angle)
        else:
            angle = -angle
        angles.append(angle)
    
    for a in angles.copy():
        if abs(a) < 5:
            angles.remove(a)

    (h,w) = img.shape[:2]
    center = (w//2, h//2)
    

    try:
        smallest_angle = min(angles, key=abs)
    except Exception as e:
        smallest_angle = 0

    if abs(smallest_angle) > 5.2:
        smallest_angle = 0

    #best_angle = min(angles, key=abs)
    print('Rotation angle :{:.3f}'.format(smallest_angle))

    if smallest_angle < 10:
        M = cv2.getRotationMatrix2D(center, -smallest_angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    else:
        M = cv2.getRotationMatrix2D(center, smallest_angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
    return rotated

#https://realpython.com/linear-regression-in-python/
def rotation_lin_regression(image_path):
    img = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    img_gs = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2GRAY)
    img_t = 1 - img_gs
    thresh = cv2.adaptiveThreshold(img_t, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 39, 3)
    img_bin = 255 - thresh
    ret, img_bin = cv2.threshold(img_bin, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    x_array = []
    y_array = []
    imgasdad, contours, hierarchy = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for contour in contours: 
        center, size, angle = cv2.minAreaRect(contour)
        if size[0] > 5 and size[1] > 5:
            x_array.append(center[0])
            y_array.append(center[1])
    
    x_array = np.array(x_array)
    y_array = np.array(y_array)
    x_array = x_array.reshape(-1, 1)
    regression = LinearRegression().fit(x_array, y_array)

    (h, w) = img.shape[0:2]
    center = (w//2, h//2)

    best_angle = regression.coef_ *180/np.pi
    print(best_angle)

    if best_angle > 2:
        M = cv2.getRotationMatrix2D(center, -best_angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    elif best_angle < 2:
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    else:
        zero_angle = 0
        M = cv2.getRotationMatrix2D(center, zero_angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
    
    return rotated
