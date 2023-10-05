import cv2 as cv
import sys
import os
import numpy as np
from matplotlib import pyplot as plt
import pydicom as dicom
import nrrd

# 1. Exploring the image, load:
starry_img = cv.imread('../imagenes/starry_night.jpg')
cv.imshow('Starry Night Original', starry_img)
# Mantener imagen
k = cv.waitKey(0)
# Guardar en la direccióm de reusltados como una nueva imagen
cv.imwrite('../Imagenesresultado/new_starry_night.png', starry_img)
starry_img_copy = starry_img.copy()

#Access the pixel in [100,100] position/seleccionar un punto en blanco
starry_img_copy[100, 100] = [255, 255, 255]
starry_img_punto = starry_img_copy[100, 100]
print(starry_img_punto)
cv.imwrite('../Imagenesresultado/new_starry_nightPunto.png', starry_img_copy)
# Access image properties/mostrar en pantalla tamaño de la imagen, canales, etc
starry_img.dtype
print(starry_img.size)
starry_img.shape
rows, cols, chans = starry_img.shape
print(rows)
# Editing and extracting ROIs (Region Of Interest)
moon = starry_img[30:190, 600:730]
new_img = starry_img
new_img[120:280, 300:430] = moon
cv.imwrite('../Imagenesresultado/new_starrySobrepuesta.png', new_img)
cv.imshow('ROI:Dos Lunas: Starry Night', new_img)
k = cv.waitKey(0)
# Resize images
scale_percent = 50
width = int(752/2)
height = int(600/2)
dim = (width, height)
resized_starry = cv.resize(starry_img, dim, interpolation=cv.INTER_CUBIC)
cv.imshow('Resized: Starry Night', resized_starry)
k = cv.waitKey(0)
cv.imwrite('../Imagenesresultado/new_starrySMALL.png', resized_starry)

# 2. Playing with colors
primary_colors_img = cv.imread('../imagenes/primary_colors.png')
cv.imshow('Primary colors Original', primary_colors_img)
k = cv.waitKey(0)
# extract the color planes B, G, R
b, g, r = cv.split(primary_colors_img)
cv.imshow('Primary colors b', b)
k = cv.waitKey(0)
cv.imwrite('../Imagenesresultado/PrimaryB.png', b)
cv.imshow('Primary colors g', g)
k = cv.waitKey(0)
cv.imwrite('../Imagenesresultado/PrimaryG.png', g)
cv.imshow('Primary colors r', r)
k = cv.waitKey(0)
cv.imwrite('../Imagenesresultado/PrimaryR.png', r)
# Merging
mergeimg = cv.merge((b, g, r))
cv.imshow('Primary colors Merge RGB', mergeimg)
# Modify color planes
b.fill(0)
bNullImg = cv.merge((b, g, r))
cv.imshow('Primary colors B NULL', bNullImg)
k = cv.waitKey(0)

# 3. Change the image type
hsvImage = cv.cvtColor(primary_colors_img, cv.COLOR_BGR2HSV)
cv.imshow('Primary colors HSV', hsvImage)
k = cv.waitKey(0)
h, s, v = cv.split(hsvImage)

cv.imshow('Primary colors H', h)
cv.imshow('Primary colors S', s)
cv.imshow('Primary colors V', v)
k = cv.waitKey(0)


# 4. Adding and blending images

x = np.uint8([250])
y = np.uint8([10])
print(cv.add(x, y))
print(x+y)
print(250+10)

# Resized blending images
ml_img = cv.imread('../imagenes/ml.png')
cv.imshow('Ml imagen Original', ml_img)
upna_logo_img = cv.imread('../imagenes/upna_logo.png')
cv.imshow('Upna Logo Original', upna_logo_img)
k = cv.waitKey(0)
resize_Cubic=cv.resize(upna_logo_img,None,fx=0.11, fy=0.11,interpolation=cv.INTER_CUBIC)
cv.imshow('Cubic:Upna resized', resize_Cubic)
resizedML = cv.resize(ml_img, None, fx=0.11, fy=0.11,interpolation=cv.INTER_NEAREST)
cv.imshow('Ml resized', resizedML)
resizedUpna = cv.resize(upna_logo_img, None, fx=0.11,fy=0.11, interpolation=cv.INTER_NEAREST)
cv.imshow('Upna resized', resizedUpna)
k = cv.waitKey(0)


# 5. InterCubir, nearest, etc resized 
orig_image = np.array([[100, 0, 130, 44, 66], [70, 144, 220, 66, 88],[22, 99, 130, 190, 12], [120, 133, 200, 100, 156], [99, 122, 55, 66, 200]], dtype=np.uint8)
originalImg = plt.imshow(orig_image, interpolation='nearest')
dim_cubic=(200,200)

resizedOriginalImg = cv.resize(orig_image, None, fx=200, fy=200, interpolation=cv.INTER_NEAREST)
cv.imshow('NEAREST:Original Image ', resizedOriginalImg)
k = cv.waitKey(0)
orig_image_lineal = cv.resize(orig_image, None, fx=200,fy=200, interpolation=cv.INTER_LINEAR)
cv.imshow('LINEAL:Original Image  ', orig_image_lineal)
k = cv.waitKey(0)
orig_image_cubic = cv.resize(orig_image, dim_cubic, interpolation=cv.INTER_CUBIC)
cv.imshow('NEAREST:Original Image ', orig_image_cubic)
k = cv.waitKey(0)

# Zero padding for images
ml_rows, ml_cols, chans = ml_img.shape
small_logo_rows, small_logo_cols, chans = resizedUpna.shape

pad_logo_img = cv.copyMakeBorder(resizedUpna, ml_rows-small_logo_rows,0, ml_cols-small_logo_cols, 0, cv.BORDER_CONSTANT, value=0)
print(ml_rows-small_logo_rows)
cv.imshow('Original Image resized lINEAASL', pad_logo_img)
k = cv.waitKey(0)

# COLOCAR IMAGEN CON FILL RED
pad_logo_img_red = cv.copyMakeBorder(resizedUpna, ml_rows-small_logo_rows,0, ml_cols-small_logo_cols, 0, cv.BORDER_CONSTANT, value=[0, 0, 250])
cv.imshow('UPNA Y ML FONDO ROJO', pad_logo_img_red)
k = cv.waitKey(0)

# Church images

church_img = cv.imread('../imagenes/church_small.jpg')
cv.imshow('Church samll image', church_img)
k = cv.waitKey(0)
# AÑadiendo 400 columnas y filas
church_img_400 = cv.copyMakeBorder(church_img, 400, 400, 400, 400, cv.BORDER_CONSTANT, value=[0, 0, 250])
cv.imshow('CHURCH 400', church_img_400)
k = cv.waitKey(0)
# Church reflejo
church_img_reflect = cv.copyMakeBorder(church_img, 400, 400, 450, 450, cv.BORDER_REFLECT)
cv.imshow('CHURCH REFLECT', church_img_reflect)
k = cv.waitKey(0)
# Church replicado
church_img_replicate = cv.copyMakeBorder(church_img, 400, 400, 450, 450, cv.BORDER_REPLICATE)
cv.imshow('CHURCH REPLICATE', church_img_replicate)
k = cv.waitKey(0)
# BORDER_WRAP
church_img_wrap = cv.copyMakeBorder(church_img, 400, 400, 450, 450, cv.BORDER_WRAP)
cv.imshow('CHURCH WRAP', church_img_wrap)
k = cv.waitKey(0)


# # Adding images cv.add

# resupna=cv.resize(upna_logo_img,[512,512])
# resml=cv.resize(ml_img,[512,512])
# add_image=cv.add(resml,resupna)
# cv.imshow('add images',add_image)
# k=cv.waitKey(0)

# #resupna1=cv.resize(upna_img,[255,255])
# add_weighted=cv.addWeighted(resml, .5,resupna, .5, 0)
# cv.imshow('weighted',add_weighted)
# k=cv.waitKey(0)

ml_ROI = ml_img 
ml_ROI[183-69:183, 20:150] = resizedUpna
cv.imshow('ROI image', ml_ROI)
cv.waitKey(0)


add_imagenes = cv.add(ml_img, pad_logo_img)
# cv.imshow('SOBREPUESTA',add_imagenes)
plt.imshow(add_imagenes)
plt.show()

k = cv.waitKey(0)
add_weighted = cv.addWeighted(ml_img, .5, pad_logo_img, .5, 0)
cv.imshow('SOBREPUESTA 2', add_weighted)
k = cv.waitKey(0)

# PARTE 6
aux = dicom.dcmread('../imagenes/knee1.dcm')
knee1_img = aux.pixel_array
cv.imshow('dst', knee1_img*128)
k = cv.waitKey(0)
aux = dicom.dcmread('../imagenes/knee2.dcm')
knee2_img = aux.pixel_array
cv.imshow('dssst', knee2_img*128)
k = cv.waitKey(0)

# knee1_img_1 = cv.normalize(knee1_img, None, alpha=0,beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
# cv.imshow('Knee Normalizada',knee1_img_1)
# k = cv.waitKey(0)

b = np.zeros(knee1_img.shape, dtype=knee1_img.dtype)
r = knee1_img
cv.imshow('Knee Green', r*128)
k = cv.waitKey(0)
g = knee2_img
cv.imshow('Knee Red', g*128)
k = cv.waitKey(0)

union = cv.merge((b*128, r*128, g*128))
cv.imshow('Knee ReD GREEN BLUE', union)
k = cv.waitKey(0)

# # Parte 2 del 6 3d y 2d
# aux = loadmat('../imagenes/mri_01.mat')
# mri_vol = aux['mri_01']
# print(mri_vol)
# mrinormalize = cv.normalize(mri_vol, None, alpha=0,beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
# print(mrinormalize)
# mri_slice = mrinormalize[:,:,6]
# cv.imshow('DATOS MRI NORMALIZADA',mri_slice)
# k = cv.waitKey(0)

# aux = loadmat('../imagenes/ct_01.mat')
# ct_vol = aux['ct_01']
# print(ct_vol)
# ctnormalize = cv.normalize(ct_vol, None, alpha=0,beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
# print(ctnormalize)
# ct_slice = ctnormalize[:,:,6]
# cv.imshow('DATOS CT NORMALIZADA',ct_slice)
# k = cv.waitKey(0)

# aux = loadmat('../imagenes/pet_01.mat')
# pet_vol = aux['pet_01']
# print(pet_vol)
# petnormalize = cv.normalize(pet_vol, None, alpha=0,beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
# print(petnormalize)
# pet_slice = petnormalize[:,:,6]
# cv.imshow('DATOS PET NORMALIZADA',pet_slice)
# k = cv.waitKey(0)


# mri_slice = mrinormalize[:,:,6]
# ct_slice = ctnormalize[:,:,7]
# pet_slice = petnormalize[:,:,4]


# resizeMri= cv.resize(mri_slice,[512,512])
# cv.imshow('Upna resized iNTER CUBUC', resizeMri)

# b=np.zeros(resizeMri.shape, dtype=resizeMri.dtype)
# g=ct_slice

# print(ct_slice.shape)

# cv.imshow('IMAGENES CEREBRO Green',r)
# k = cv.waitKey(0)
# r=resizeMri
# cv.imshow('IMAGENES CEREBRO Red',g)
# k = cv.waitKey(0)

# union=cv.merge((b,r,g))
# cv.imshow('IMAGENES CEREBRO ReD GREEN BLUE',union)
# k = cv.waitKey(0)
