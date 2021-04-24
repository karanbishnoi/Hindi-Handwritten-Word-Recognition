import numpy as np
import pandas as pd
import os
import keras
import cv2




def image_clear(image):
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
    thresh = cv2.bitwise_not(thresh)
    thresh = cv2.GaussianBlur(thresh, (3,3), 1)
    kernel1 = np.ones((3,3), np.uint8)
    thresh = cv2.erode(thresh, kernel1, iterations = 1)
    ret,thresh=cv2.threshold(thresh,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel2 = np.ones((3,3), np.uint8)
    thresh = cv2.dilate(thresh, kernel2, iterations = 1)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, None, None, None, 8, cv2.CV_32S)
    areas = stats[1:,cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if areas[i] >= 100:
            result[labels == i + 1] = 255
    return result




def image_skrew_correction(thresh):
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle > 45:
        angle = -angle+90
    else:
        angle = -angle
    (h, w) = thresh.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(thresh, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated = cv2.bitwise_not(rotated)
    return rotated




def remove_header(image):
    row = []
    h, w = image.shape
    for i in range(0, int(h*0.75)):
        cnt = list(image[i]).count(0)
        row.append(cnt)
    row_id = row.index(max(row))
    row_cnt = row[row_id]
    for i in range(0, row_id):
        image[i] = 255
    for i in range(0, int((h-row_id)*0.2)):
        cnt = list(image[row_id+i]).count(0)
        if cnt >= row_cnt*0.2:
            image[row_id+i] = 255
        else:
            break
    return image




def getMeanArea(contours):
    meanArea=0
    for contour in contours:
        meanArea+=cv2.contourArea(contour)
    meanArea=(meanArea)/len(contours)
    return meanArea



def character_extraction(original,image):
    image = cv2.bitwise_not(image)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    meanArea = getMeanArea(contours)
    count=0
    coords=[]
    coordinates = []
    for contour in contours:
        (x,y,w,h)=cv2.boundingRect(contour)
        if cv2.contourArea(contour)>0.5*meanArea:
            coordinates.append((x,y,w,h))
    coordinates.sort(key = lambda x: x[0])
    n = len(coordinates)
    coordinates_final = []
    index = 0
    while(index < n-1):
        x1 = coordinates[index+1][0]
        x2 = coordinates[index][0]+coordinates[index][2]
        if(x2 > x1):
            x = coordinates[index][0]
            y = coordinates[index][1]
            w = coordinates[index+1][0]+coordinates[index+1][2]-coordinates[index][0]
            h = max(coordinates[index][3],coordinates[index][3])
            coordinates_final.append((x,y,w,h))
            index += 1
        else:
            coordinates_final.append(coordinates[index])
        index += 1
    if index == n-1:
        coordinates_final.append(coordinates[index])
    for coordis in coordinates_final:
        (x,y,w,h) = coordis
        if w / h > 1.4:
            half_width = int(w / 2)
            coords.append((x, y, half_width, h))
            coords.append((x + half_width, y, half_width, h))
            count=count+2
        else:
            coords.append((x, y, w, h))
            count=count+1
    coords.sort(key = lambda x: x[0])
    output_array = []
    for i in range(count):
        x = coords[i][0]
        y = coords[i][1]
        z = min(15,y)
        y -= z
        w = coords[i][2]
        h = coords[i][3]+z
        result = original[y:y+h,x:x+w]
        result = cv2.resize(result,(100,100))
        outputImage = cv2.copyMakeBorder(
                         result, 
                         5, 
                         5, 
                         5, 
                         5, 
                         cv2.BORDER_CONSTANT, 
                         value=255
                      )
        filename='character'+str(i+1)+'.jpg'
        cv2.imwrite(filename,cv2.bitwise_not(outputImage))
        cv2.imshow(str(i+1),outputImage)
        output_array.append(filename)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return output_array

def predict(image_path):
    image = cv2.imread(image_path, 0)
    h, w = image.shape
    image = cv2.resize(image,(int(w/2), int(h/2)))
    # thresh is image with threshold applied any image is converted to binary
    thresh = image_clear(image.copy())
    # img is the converted image with horizontal alignment 
    img = image_skrew_correction(thresh.copy())
    # final_img is the image with header line deleted
    final_img = remove_header(img.copy())
    cv2.imshow('final',final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    image_paths = character_extraction(img,final_img.copy())
    output = []
    for i in image_paths:
        m = []
        image = cv2.imread(i,cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image,(32,32))
        image = np.reshape(image,(32,32,1))/255
        m.append(image)
        m = np.array(m)
        y_classes = np.argmax(model.predict(m))
        output.append(catergories[y_classes])
    return output


model = keras.models.load_model('model_hindi.hdf5')
CATEGORIES = ['क','घ','च','ज','ञ','ट','ठ','ड','त','द','ध','न',
              'प','फ','ब','म','र','ल','व','ष','स','ह','क्ष','त्र','ज्ञ']
catergories = np.array(CATEGORIES)

def test():
    #Enter filenames to be tested in image_paths after adding them to this folder
    image_paths = ['sample_images/t8.jpg']
    for i,image_path in enumerate(image_paths):
        answer = predict(image_path)
        print(''.join(answer))# will be the output string

if __name__=='__main__':
    test()