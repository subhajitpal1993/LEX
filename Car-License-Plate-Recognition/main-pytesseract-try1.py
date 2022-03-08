import cv2
import numpy as np
from skimage.filters import threshold_local
import tensorflow as tf
from skimage import measure
import imutils
import pytesseract

def sort_cont(character_contours):
    """
    To sort contours from left to right
    """
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in character_contours]
    (character_contours, boundingBoxes) = zip(*sorted(zip(character_contours, boundingBoxes),
                                                      key=lambda b: b[1][i], reverse=False))
    return character_contours


def segment_chars(plate_img, fixed_width):
    """
    extract Value channel from the HSV format of image and apply adaptive thresholding
    to reveal the characters on the license plate
    """
    V = cv2.split(cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV))[2]

    T = threshold_local(V, 29, offset=15, method='gaussian')

    thresh = (V > T).astype('uint8') * 255

    thresh = cv2.bitwise_not(thresh)

    # resize the license plate region to a canoncial size
    plate_img = imutils.resize(plate_img, width=fixed_width)
    thresh = imutils.resize(thresh, width=fixed_width)
    bgr_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # perform a connected components analysis and initialize the mask to store the locations
    # of the character candidates
    labels = measure.label(thresh,background=0)

    charCandidates = np.zeros(thresh.shape, dtype='uint8')

    # loop over the unique components
    characters = []
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask to display only connected components for the
        # current label, then find contours in the label mask
        labelMask = np.zeros(thresh.shape, dtype='uint8')
        labelMask[labels == label] = 255

        cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        # cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # ensure at least one contour was found in the mask
        if len(cnts) > 0:

            # grab the largest contour which corresponds to the component in the mask, then
            # grab the bounding box for the contour
            c = max(cnts, key=cv2.contourArea)
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

            # compute the aspect ratio, solodity, and height ration for the component
            aspectRatio = boxW / float(boxH)
            solidity = cv2.contourArea(c) / float(boxW * boxH)
            heightRatio = boxH / float(plate_img.shape[0])

            # determine if the aspect ratio, solidity, and height of the contour pass
            # the rules tests
            keepAspectRatio = aspectRatio < 1.0
            keepSolidity = solidity > 0.15
            keepHeight = heightRatio > 0.5 and heightRatio < 0.95

            # check to see if the component passes all the tests
            if keepAspectRatio and keepSolidity and keepHeight and boxW > 14:
                # compute the convex hull of the contour and draw it on the character
                # candidates mask
                hull = cv2.convexHull(c)

                cv2.drawContours(charCandidates, [hull], -1, 255, -1)

    contours, hier = cv2.findContours(charCandidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sort_cont(contours)
        addPixel = 4  # value to be added to each dimension of the character
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if y > addPixel:
                y = y - addPixel
            else:
                y = 0
            if x > addPixel:
                x = x - addPixel
            else:
                x = 0
            temp = bgr_thresh[y:y + h + (addPixel * 2), x:x + w + (addPixel * 2)]

            characters.append(temp)
        return characters
    else:
        return None

def predict_char(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    ctrs, hier = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    img_contour = img.copy()
    for i in range(len(sorted_ctrs)):
        area = cv2.contourArea(sorted_ctrs[i])
        if 100 < area < 10000:
            cv2.drawContours(img_contour, sorted_ctrs, i, (0, 0, 255), 2)

    detected = ""
    #orted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * img.shape[1] )
    for c in sorted_ctrs:

        x, y, w, h = cv2.boundingRect(c)
        ratio = h/w
        area = cv2.contourArea(c)
        base = np.ones(thresh.shape, dtype=np.uint8)
        
        if ratio > 0.9 and 100 < area < 10000:
            base[y:y+h, x:x+w] = thresh[y:y+h, x:x+w]
            
            segment = cv2.bitwise_not(base)

#             print(segment.tolist().index(1))
            custom_config = r'-l eng --oem 3 --psm 10 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890-." '
            c = pytesseract.image_to_string(segment, config=custom_config)
   
            detected = detected + c

    detected=detected.replace("\n", "")
    
    return detected


class PlateFinder:
    def __init__(self):
        self.min_area = 6500  # minimum area of the plate # earlier 7000
        self.max_area = 30000  # maximum area of the plate

        self.element_structure = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(22, 3))

    def preprocess(self, input_img):
        imgBlurred = cv2.GaussianBlur(input_img, (7, 7), 0)  # old window was (5,5)
        gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)  # convert to gray
        sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)  # sobelX to get the vertical edges
        ret2, threshold_img = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        element = self.element_structure
        morph_n_thresholded_img = threshold_img.copy()
        cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_n_thresholded_img)
        return morph_n_thresholded_img

    def extract_contours(self, input_img,after_preprocess):
        contours, _ = cv2.findContours(after_preprocess, mode=cv2.RETR_EXTERNAL,
                                                        method=cv2.CHAIN_APPROX_SIMPLE)
        
        #for contour in contours:
        #    area = int(cv2.contourArea(contour))
        #    x, y, w, h = cv2.boundingRect(contour)
        #    ratio=int(w/h)
        #    cv2.drawContours(input_img, contour, -1, (0, 230, 255), 1)
        #    cv2.putText(input_img, str(ratio), (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        #    cv2.putText(input_img, str(area), (x, y-30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
        #    cv2.imshow('Contours', input_img)
        #    if cv2.waitKey(1) & 0xFF == ord('q'):
        #        break          
        return contours

    def clean_plate(self, plate):
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)  # index of the largest contour in the area array

            max_cnt = contours[max_index]
            max_cntArea = areas[max_index]
            x, y, w, h = cv2.boundingRect(max_cnt)
            rect = cv2.minAreaRect(max_cnt)
            rotatedPlate = plate
            if not self.ratioCheck(max_cntArea, rotatedPlate.shape[1], rotatedPlate.shape[0]):
                return plate, False, None
            return rotatedPlate, True, [x, y, w, h]
        else:
            return plate, False, None



    def check_plate(self, input_img, contour):
        min_rect = cv2.minAreaRect(contour)

        x, y, w, h = cv2.boundingRect(contour)


        after_validation_img = input_img[y:y + h, x:x + w]

        after_clean_plate_img, plateFound, coordinates = self.clean_plate(after_validation_img)
        if plateFound:
            characters_on_plate = self.find_characters_on_plate(after_clean_plate_img)
            #if (characters_on_plate is not None):
            #    print(len(characters_on_plate))
            if (characters_on_plate is not None and len(characters_on_plate) == 8):
                #print(len(characters_on_plate))
                x1, y1, w1, h1 = coordinates
                coordinates = x1 + x, y1 + y
                after_check_plate_img = after_clean_plate_img
                return after_check_plate_img, characters_on_plate, coordinates
        return None, None, None



    def find_possible_plates(self, input_img):
        """
        Finding all possible contours that can be plates
        """
        plates = []
        self.char_on_plate = []
        self.corresponding_area = []

        self.after_preprocess = self.preprocess(input_img)
        possible_plate_contours = self.extract_contours(input_img,self.after_preprocess)

        for cnts in possible_plate_contours:
            plate, characters_on_plate, coordinates = self.check_plate(input_img, cnts)
            if plate is not None:
                plates.append(plate)
                self.char_on_plate.append(characters_on_plate)
                self.corresponding_area.append(coordinates)

        if (len(plates) > 0):
            return plates
        else:
            return None

    def find_characters_on_plate(self, plate):

        charactersFound = segment_chars(plate, 400)
                
        if charactersFound:
            return charactersFound

    # PLATE FEATURES
    def ratioCheck(self, area, width, height):
        min = self.min_area
        max = self.max_area

        ratioMin = 1.5
        ratioMax = 6

        ratio = float(width) / float(height)
        if ratio < 1:
            ratio = 1 / ratio

        # print("AREA :"+str(area))
        if (area < min or area > max) or (ratio < ratioMin or ratio > ratioMax):
            return False
        return True



recognized_list=[]

def listToString(s): 
    str1 = " "    
    return (str1.join(s))      
    
      
    
if __name__ == "__main__":
    findPlate = PlateFinder()

    # Initialize the Neural Network
#     model = NeuralNetwork()
    count=0
    cap = cv2.VideoCapture('test.MOV')
    while (cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            count+=1
            #img = cv2.resize(img, (960, 540))            
            cv2.imshow('original video', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # cv2.waitKey(0)
            possible_plates = findPlate.find_possible_plates(img)

            #after_preprocess = findPlate.preprocess(img)
            #extract_contours=findPlate.extract_contours(after_preprocess)
            #cv2.drawContours(after_preprocess, extract_contours, -1, (0, 255, 0), 1)
            #cv2.imshow('Contours', after_preprocess)
            #if cv2.waitKey(5) & 0xFF == ord('q'):
            #    break

            if possible_plates is not None:
                for i, p in enumerate(possible_plates):
                    #chars_on_plate = findPlate.char_on_plate[i]
                    cv2.imwrite('C:/Users/subha/MY_PROJECTS/Car-License-Plate-Recognition/frames/frame%d.jpg'%count,p)
                    recognized_plate=predict_char(p)
                    
                    print(recognized_plate[0])
                    if recognized_plate is not None :                    
                        if (len(recognized_plate)==8) and (recognized_plate[0] in ("1","2","3","4","5","6","7","8","9","0")) and (recognized_plate[1] in ("1","2","3","4","5","6","7","8","9","0")) and (recognized_plate[2] not in ("1","2","3","4","5","6","7","8","9","0")) and (recognized_plate[7] in ("1","2","3","4","5","6","7","8","9","0")) and (recognized_plate[3] in ("1","2","3","4","5","6","7","8","9","0")) and (recognized_plate[4] in ("1","2","3","4","5","6","7","8","9","0")) and (recognized_plate[5] in ("1","2","3","4","5","6","7","8","9","0")) and (recognized_plate[6] in ("1","2","3","4","5","6","7","8","9","0")):                    

                            recognized_list.append(recognized_plate)
                            recognized_list=list(set(recognized_list))
                            print(recognized_list, end=" ")
                                #cv2.putText(p, recognized_plate, (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)                   


                        cv2.imshow('plate', p)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break


        else:
            break
    
    
    
    
    print("The list of CAR license plates read by ENGINE :"+ listToString(recognized_list))
    cap.release()
    cv2.destroyAllWindows()
