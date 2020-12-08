import cv2
import numpy as np

#some function 
def draw_line(img, lines, color = [0,255,0], thickness = 10):
    #in case you dont want to draw, then dont
    draw_right = True
    draw_left = True
    #find the line's slope, and only care about the one that have   0.5< abs(slope) <0.8
    slope_threshold = 0.5
    slopes = []
    new_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0] #line[0] = [[x1 y1 x2 y2]]

        if x2-x1 == 0. : 
            slope = 999.
        else :
            slope = (y2-y1)/(x2-x1)

        if abs(slope) > slope_threshold:
            slopes.append(slope)
            new_lines.append(line)
    lines = new_lines
    #seperate the lines into left and right lines
    right_line = []
    left_line = []
    for i,line in enumerate(lines):
        x1,y1,x2,y2 = line[0]
        img_x_center = img.shape[1]/2
        if slopes[i] >0 and x1 > img_x_center and x2 > img_x_center:
            right_line.append(line)
        elif slopes[i] <0 and x1 < img_x_center and x2 < img_x_center:
            left_line.append(line)
    #tìm phương trình đường thẳng đi qua 2 điểm để vẽ line 
    #right line
    right_x = []
    right_y = []
    for line in right_line:
        x1, y1, x2, y2 = line[0]
        right_x.append(x1)
        right_x.append(x2)
        right_y.append(y1)
        right_y.append(y2)
    if len(right_x) >0:
        #y = mx+b
        right_m, right_b = np.polyfit(right_x, right_y, 1)
    else: 
        right_m, right_b = 1,1
        draw_right = False
    #left line
    left_x = []
    left_y = []
    for line in left_line:
        x1,y1,x2,y2 = line[0]
        left_x.append(x1)
        left_x.append(x2)
        left_y.append(y1)
        left_y.append(y2)
    if len(left_x) > 0:
        #y = mx + b
        left_m, left_b = np.polyfit(left_x,left_y,1)
    else:
        left_m,left_b = 1,1
        draw_left = False
    #y = mx + b -->x = (y-b)/ m 
    y1 = img.shape[0]
    y2 = img.shape[0] *0.7
    right_x1 = (y1 - right_b) / right_m
    right_x2 = (y2 - right_b) / right_m
    left_x1 = (y1 - left_b) / left_m
    left_x2 = (y2 - left_b) / left_m
    y1 = int(y1)
    y2 = int(y2)
    right_x1 = int(right_x1)
    right_x2 = int(right_x2)
    left_x1 = int(left_x1)
    left_x2 = int(left_x2)
    #center point
    c_x1 = int((right_x1+left_x1)/2)
    c_x2 = int((right_x2+left_x2)/2)

    #finally, draw line
    if draw_right:
        cv2.line(img, (right_x1,y1), (right_x2,y2), color, thickness)
        cv2.line(img, (c_x1, y1), (c_x2, y2), color, 2)
    if draw_left:
        cv2.line(img, (left_x1,y1), (left_x2,y2), color, thickness)

def filter(img):
    #white line in bgr color space
    low_white = np.array([200,200,200])
    up_white = np.array([255,255,255])
    mask_white = cv2.inRange(img, low_white, up_white)
    white_image = cv2.bitwise_and(img, img, mask = mask_white)

    #yellow line in hsv color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_yellow = np.array([15,94,100])
    up_yellow = np.array([37, 255, 255])
    mask_yellow = cv2.inRange(hsv, low_yellow, up_yellow)
    yellow_image = cv2.bitwise_and(img, img, mask= mask_yellow)

    #combine two image together
    #mask = cv2.bitwise_and(mask_white, mask_yellow)
    img2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0)

    return img2

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    #mask is a black image have the same shape with the original image
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        #the color of the roi in the mask (it hard to explain, just show the mask to see)
        roi_color = [255,] *channel_count 
    else:
        roi_color = 255
    cv2.fillPoly(mask, vertices, roi_color)

    img = cv2.bitwise_and(img, mask)
    return img

def canny(img,threshold1, threshold2):
    return cv2.Canny(img, threshold1, threshold2)

def draw_line_v2(img, lines, color = [255,0,0,], thickness = 10):
    #calculate the slope and then only take one that bigger than 0.5 and smaller than 20
    right_line_coordinates = []
    left_line_coordinates = []
    #calculate slope, filter slope, classify lines into right lines and left lines
    for line in lines:
        x1, y1, x2, y2 = line[0]

        #calculate slope
        if (x2-x1) == 0. : slope = 999.
        else: slope = (y2-y1)/ (x2-x1)

        #filter  out slope that bigger than 0.5 and smaller than 20
        # and then classify lines in to right lines and left lines
        if abs(slope) > 0.5 and  abs(slope) < 20: 
            if slope <0 : left_line_coordinates.append([[x1, y1, x2, y2]])
            elif slope > 0. : right_line_coordinates.append([[x1, y1, x2, y2]])

    #using np.polyfit to find out the best fit line
    #for the right lines
    right_x = []
    right_y = []
    for  coordinates in right_line_coordinates:
        x1, y1, x2, y2 = coordinates[0]
        right_x.append(x1)
        right_x.append(x2)
        right_y.append(y1)
        right_y.append(y2)
    if len(right_x):
        # y = ax + b
        right_a, right_b = np.polyfit(right_x, right_y, 1)
    else: 
        right_a, right_b = 1, 1
    
    #for the left lines
    left_x = []
    left_y = []
    for coordinates in left_line_coordinates:
        x1, y1, x2, y2 = coordinates[0]
        left_x.append(x1)
        left_x.append(x2)
        left_y.append(y1)
        left_y.append(y2)
    if len(left_x):
        # y = ax + b
        left_a,  left_b = np.polyfit(left_x, left_y, 1)
    else : 
        lefa, left_b = 1, 1

    #initiate x, y (y=ax+b --> x = (y-b)/a)
    y1 = img.shape[0]
    y2 = img.shape[0] * 0.6
    right_x1 = (y1 - right_b)/ right_a
    right_x2 = (y2 - right_b)/ right_a
    left_x1 = (y1 - left_b)/ left_a
    left_x2 = (y2 - left_b)/ left_a

    #center point 
    center_point_x2 = int((right_x2 + left_x2)/2)
    center_point_x1 = int((right_x1 + left_x1)/2)
    y1 = int(y1)
    y2 = int(y2)
    right_x1 = int(right_x1)
    right_x2 = int(right_x2)
    left_x1 = int(left_x1)
    left_x2 = int(left_x2) 
    #print(center_point_x1, center_point_x2)
    cv2.circle(img, (center_point_x1, y1), 40, color, -1)
    cv2.circle(img, (center_point_x2, y2), 30, color, -1)
    cv2.line(img, (right_x1,y1), (right_x2,y2), color, thickness)


#let test it out

cap = cv2.VideoCapture('original_mp4\solidWhiteRight.mp4')

while(1):
    ret, ori_img = cap.read()
    if not ret:
        cap = cv2.VideoCapture('original_mp4\solidWhiteRight.mp4')
        continue
    vertices = [
    (0,ori_img.shape[0]),
    (ori_img.shape[1]/2, ori_img.shape[0]/2),
    (ori_img.shape[1], ori_img.shape[0])
    ]
    #img = ori_img.copy()
    img = filter(ori_img)
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, 350,750)
    roi = region_of_interest(edges, np.array([vertices], np.int32))
    lines = cv2.HoughLinesP(roi, 2, np.pi/180, 150, minLineLength=10, maxLineGap= 20)
    if lines is not None:
        draw_line(ori_img, lines)
    cv2.imshow('frame', ori_img)

    if cv2.waitKey(25)&0xff == 27:
        break


cap.release()
cv2.destroyAllWindows()
