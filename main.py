# Linear Equation
# z = x1 + x2
# x1 + 3 * x2 < 3
# 2 * x1 + x2 < 2
# 3 * x1 + x2 < 3

import numpy as np
import cv2

SCREEN_PIXELS = 512
AXIS_UNIT = 16
ARROW_PIXELS = 10
# multiplication
MULTIP = int(SCREEN_PIXELS / AXIS_UNIT)

AXIS_COLOR = (255, 0, 0)  # BGR
AXIS_THICKNESS = 2
EQ_COLOR = (0, 0, 255)  # BGR
EQ_THICKNESS = 1

# Convert point to pixel


def convert(point, type):
    if point == 0:
        pixel = SCREEN_PIXELS / 2

    elif type == 'x1':
        if point > 0:
            pixel = (AXIS_UNIT / 2 - point) * MULTIP
        elif point < 0:
            pixel = (-point * MULTIP) + (SCREEN_PIXELS / 2)

    elif type == 'x2':
        if point > 0:
            pixel = (point * MULTIP) + (SCREEN_PIXELS / 2)
        elif point < 0:
            pixel = (AXIS_UNIT / 2 + point) * MULTIP

    return int(pixel)


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C


def drawInequality(x1_factor, x2_factor, c):
    x1 = int(-c / x1_factor)
    x2 = int(-c / x2_factor)

    start_pixels = (convert(0, 'x2') - (10 * x2 * MULTIP),
                    convert(x1, 'x1') - (10 * x1 * MULTIP))
    end_pixels = (convert(x2, 'x2') + (10 * x2 * MULTIP),
                  convert(0, 'x1') + (10 * x1 * MULTIP))

    cv2.line(img, start_pixels, end_pixels, EQ_COLOR, EQ_THICKNESS)

    return line([0,x1], [x2,0])

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False


# Create a black image
img = np.zeros((SCREEN_PIXELS, SCREEN_PIXELS, 3), np.uint8)

# x1 axis
cv2.line(img, (int(SCREEN_PIXELS/2), 0), (int(SCREEN_PIXELS/2), SCREEN_PIXELS),
         AXIS_COLOR, AXIS_THICKNESS)
cv2.line(img, (int(SCREEN_PIXELS/2), 0),
         (int(SCREEN_PIXELS/2) - ARROW_PIXELS, ARROW_PIXELS), AXIS_COLOR, AXIS_THICKNESS)
cv2.line(img, (int(SCREEN_PIXELS/2), 0), (int(SCREEN_PIXELS/2) + ARROW_PIXELS, ARROW_PIXELS),
         AXIS_COLOR, AXIS_THICKNESS)
cv2.line(img, (int(SCREEN_PIXELS/2), SCREEN_PIXELS),
         (int(SCREEN_PIXELS/2) + ARROW_PIXELS, SCREEN_PIXELS - ARROW_PIXELS), AXIS_COLOR, AXIS_THICKNESS)
cv2.line(img, (int(SCREEN_PIXELS/2), SCREEN_PIXELS),
         (int(SCREEN_PIXELS/2) - ARROW_PIXELS, SCREEN_PIXELS - ARROW_PIXELS), AXIS_COLOR, AXIS_THICKNESS)
cv2.putText(img=img, text="X1", org=(int((SCREEN_PIXELS/2) + (2*ARROW_PIXELS)), ARROW_PIXELS),
            fontFace=cv2.QT_FONT_NORMAL, fontScale=0.3, color=(0, 255, 0), thickness=1)

# x2 axis
cv2.line(img, (0, int(SCREEN_PIXELS/2)), (SCREEN_PIXELS, int(SCREEN_PIXELS/2)),
         AXIS_COLOR, AXIS_THICKNESS)
cv2.line(img, (SCREEN_PIXELS, int(SCREEN_PIXELS/2)), (SCREEN_PIXELS-ARROW_PIXELS, int(SCREEN_PIXELS/2)-ARROW_PIXELS),
         AXIS_COLOR, AXIS_THICKNESS)
cv2.line(img, (SCREEN_PIXELS, int(SCREEN_PIXELS/2)), (SCREEN_PIXELS-ARROW_PIXELS, int(SCREEN_PIXELS/2)+ARROW_PIXELS),
         AXIS_COLOR, AXIS_THICKNESS)
cv2.line(img, (0, int(SCREEN_PIXELS/2)), (ARROW_PIXELS, int(SCREEN_PIXELS/2)-ARROW_PIXELS),
         AXIS_COLOR, AXIS_THICKNESS)
cv2.line(img, (0, int(SCREEN_PIXELS/2)), (ARROW_PIXELS, int(SCREEN_PIXELS/2)+ARROW_PIXELS),
         AXIS_COLOR, AXIS_THICKNESS)
cv2.putText(img=img, text="X2", org=(SCREEN_PIXELS-(2*ARROW_PIXELS), int(SCREEN_PIXELS/2) -
            (2*ARROW_PIXELS)), fontFace=cv2.QT_FONT_NORMAL, fontScale=0.3, color=(0, 255, 0), thickness=1)

# numbers for x
for i in range(int(-AXIS_UNIT/2 + 1), int(AXIS_UNIT/2)):
    cv2.putText(img=img, text=str(i), org=(convert(i, 'x2'), convert(0, 'x1')),
                fontFace=cv2.QT_FONT_NORMAL, fontScale=0.3, color=(0, 255, 0), thickness=1)
# numbers for y
for i in range(int(-AXIS_UNIT/2 + 1), int(AXIS_UNIT/2)):
    cv2.putText(img=img, text=str(i), org=(convert(0, 'x2'), convert(i, 'x1')),
                fontFace=cv2.QT_FONT_NORMAL, fontScale=0.3, color=(0, 255, 0), thickness=1)

# Linear Equation
# z = x1 + x2

# x1 + 3 * x2 < 3
l1 = drawInequality(1, 3, -3)
# 2 * x1 + x2 < 2
l2 = drawInequality(2, 1, -2)
# 3 * x1 + x2 < 3
l3= drawInequality(3, 1, -3)

x1_axis = line([0, int(-AXIS_UNIT/2)], [0, int(AXIS_UNIT/2)])
x2_axis = line([int(-AXIS_UNIT/2), 0], [int(AXIS_UNIT/2), 0])

in1 = intersection(l1, l2)
in2 = intersection(l1, l3)
in3 = intersection(l2, l3)
in_each_other = [in1, in2, in3]
min_x1_1 = min(map(lambda a: a[1], in_each_other))
min_point1 = list(filter(lambda a: a[1] == min_x1_1, in_each_other))[0]

in4 = intersection(l1, x1_axis)
in5 = intersection(l2, x1_axis)
in6 = intersection(l3, x1_axis)
in_x1_axis = [in4, in5, in6]
min_x1_2 = min(map(lambda a: a[1], in_x1_axis))
min_point2 = list(filter(lambda a: a[1] == min_x1_2, in_x1_axis))[0]

in7 = intersection(l1, x2_axis)
in8 = intersection(l2, x2_axis)
in9 = intersection(l3, x2_axis)
in_x2_axis = [in7, in8, in9]
min_x1_3 = min(map(lambda a: a[1], in_x2_axis))
min_point3 = list(filter(lambda a: a[1] == min_x1_3, in_x2_axis))[0]

min_point4 = (0,0)

# Polygon corner points coordinates
pts = np.array([
    [convert(min_point1[0],'x2'), convert(min_point1[1],'x1')],
    [convert(min_point2[0],'x2'), convert(min_point2[1],'x1')],
    [convert(min_point4[0],'x2'), convert(min_point4[1],'x1')],
    [convert(min_point3[0],'x2'), convert(min_point3[1],'x1')],
], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.fillPoly(img, [pts], (0, 255, 0))

# Find min and max points
min_points = [min_point1, min_point2, min_point3, min_point4]
z_min = min_point1[0] + min_point1[1]
z_min_point = min_point1
z_max = min_point1[0] + min_point1[1]
z_max_point = min_point1
for each in min_points:
    z = each[0] + each[1]
    if z < z_min:
        z_min = z
        z_min_point = each
    if z > z_max:
        z_max = z
        z_max_point = each

print("MIN ve MAX değerleri (x2, x1)")
print("MIN:", z_min_point)
print("MAX:", z_max_point)

# Displaying the image
while(1):
    cv2.imshow('Yoneylem', img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()
