import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import statistics as stat
import pandas as pd
import numpy as np

img = cv.imread("Melanoma_detection\\Dataset\\Dataset\\Big_dir\\ISIC_0024308.jpg")
red = []
gr = []
bl = []
x = []
for i in range(img.shape[0]):
    for l in range(img.shape[1]):
        gr.append(int(img[i][l][0]))
        bl.append(int(img[i][l][1]))
        red.append(int(img[i][l][2]))
        x.append(0)
       
        
plt.scatter(np.linspace(0, len(red), len(red)), gr, s=0.1,c='g')        
plt.scatter(np.linspace(0, len(red), len(red)), bl, s=0.1,c='b')        
plt.scatter(np.linspace(0, len(red), len(red)), red, s=0.1, c='r')
#plt.plot(x, color='red')
plt.grid()

red_low, gr_low, bl_low = [], [], []
medmean_r = stat.mean([stat.median(red), min(red)])
medmean_g = stat.mean([stat.median(gr), min(gr)])
medmean_b = stat.mean([stat.median(bl), min(bl)])
for i in range(min(len(red), len(gr), len(bl))):
    if red[i] < medmean_r:
        red_low.append(red[i])
    if gr[i] < medmean_g:
        gr_low.append(gr[i])
    if bl[i] < medmean_b:
        bl_low.append(bl[i])
lowmed_r = stat.median(red_low)
lowmed_g = stat.median(gr_low)
lowmed_b = stat.median(bl_low)
highmed_r = stat.median(red)+25
highmed_g = stat.median(gr)+25
highmed_b = stat.median(bl)+25
if highmed_r > 255:
    highmed_r = 255
elif highmed_g > 255:
    highmed_g == 255
elif highmed_b > 255:
    highmed_b = 255
min_p = (stat.mean([lowmed_g, highmed_g]), stat.mean([lowmed_b, highmed_b]), stat.mean([lowmed_r, highmed_r]))
max_p = (highmed_g, highmed_b, highmed_r)

print(min_p, max_p)

img_g = cv.inRange(img, min_p, max_p)
cv.imshow('img', img_g)

gik = cv.cvtColor(img, cv.COLOR_BGR2HSV)
lower_skin = np.array(min_p, dtype=np.uint8)
upper_skin = np.array(max_p, dtype=np.uint8)
skin_mask = cv.inRange(img, lower_skin, upper_skin)
skin = cv.bitwise_and(img, img, mask=skin_mask)
skin = cv.GaussianBlur(skin, (5, 5), 0)
alpha = -3  # Контраст
beta = -100  # Яркость
notflor = cv.convertScaleAbs(skin, alpha=alpha, beta=beta)

#Создание фигуры на белом фоне
hsv = cv.cvtColor(notflor, cv.COLOR_BGR2HSV)
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 25, 255])
mask = cv.inRange(hsv, lower_white, upper_white)
inverse_mask = cv.bitwise_not(mask)
red_image = np.zeros_like(notflor)
red_image[:] = [255, 0, 0]  
figure = cv.bitwise_or(red_image, red_image, mask=inverse_mask)
figure = cv.bitwise_or(figure, notflor, mask=mask)

# Определение контуров
gray = cv.cvtColor(figure, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(gray, 200, 255, cv.THRESH_BINARY_INV)
contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

if contours:
    largest_contour = max(contours, key=cv.contourArea)
    perimeter = cv.arcLength(largest_contour, True)
    approx = cv.approxPolyDP(largest_contour, 0.02 * perimeter, True)
    area = cv.contourArea(largest_contour)
    circularity = (4 * np.pi * area) / (perimeter ** 2)

    print(f'Ровность здоровой родинки: {circularity:.4f}')
    cv.drawContours(figure, [largest_contour], -1, (0, 255, 0), 3)
    cv.drawContours(figure, [approx], -1, (255, 0, 0), 3)
else:
    print("Контуры родинки не найдены.")
    
    # Функция для анализа симметрии
def calculate_symmetry(image):
    # Преобразование в градации серого
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Нахождение контуров
    contours, _ = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return 0, 0, 0

    # Предполагаем, что первые контуры - это интересующий нас объект
    cnt = contours[0]

    # Центр масс контура
    M = cv.moments(cnt)
    if M['m00'] == 0:
        return 0, 0, 0

    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])

    # Создание зеркального изображения относительно вертикальной и горизонтальной оси
    mirror_vertical = cv.flip(image, 1)
    mirror_horizontal = cv.flip(image, 0)

    # Нахождение разности между оригиналом и зеркальным
    diff_vertical = cv.absdiff(image, mirror_vertical)
    diff_horizontal = cv.absdiff(image, mirror_horizontal)

    # Подсчет количества ненулевых пикселей
    non_zero_vertical = np.count_nonzero(diff_vertical)
    non_zero_horizontal = np.count_nonzero(diff_horizontal)

    # Определение симметрии
    symmetry_vertical = 1 - (non_zero_vertical / image.size)
    symmetry_horizontal = 1 - (non_zero_horizontal / image.size)

    return symmetry_vertical, symmetry_horizontal, (cX, cY)

print(calculate_symmetry(img))
cv.imshow('Not melanoma contours', figure)
cv.waitKey(0)
cv.destroyAllWindows()
plt.show()