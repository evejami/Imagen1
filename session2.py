import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import os
# Getting the path for this file
project_path = os.path.dirname(__file__)
# Changing out current working directory. In this way, we can work with
# relative paths
os.chdir(project_path)

head_img = cv.imread("../imagenes/mri_head_sag.jpg")
cv.imshow('Mri head Image', head_img)
k = cv.waitKey(0)

mri_head_grayColor = cv.cvtColor(head_img, cv.COLOR_BGR2GRAY)
cv.imshow('Color Gray MRI head mri_head_grayColor', mri_head_grayColor)
k = cv.waitKey(0)
print("Estamos seguros que tiene color al mostrar los tres canel, r,g,b", head_img.shape)

# 2. Calculate the histogram:
# ravel_v=mri_head_grayColor.ravel()
# Primer metodo
# plt.hist(ravel_v, bins=256, range=[0,256], density=True)
# Segundo método de histograma
histograma = cv.calcHist([mri_head_grayColor], [0], None, [256], [0, 256])
plt.plot(histograma)
plt.title('Histograma de la imagen')
plt.show()
# Tercer metodo de histograma
mri_head_grayColor.flatten()


# 3. Cumulative histogram
# cdf = histograma.cumsum()
# cdf_scaled = cdf*histograma.max()/cdf.max()
# plt.plot(histograma, color='r')
# plt.plot(cdf_scaled, color='b')
# plt.title('Histogram and cdf with numpy')
# plt.show()

starry_img = cv.imread('../imagenes/starry_night.jpg')
# canal_rojo = starry_img[:, :, 2]
# canal_verde = starry_img[:, :, 1]
# canal_azul = starry_img[:, :, 0]
# histogramar = cv.calcHist([canal_rojo], [0], None, [256], [0, 256])
# histogramab = cv.calcHist([canal_azul], [0], None, [256], [0, 256])
# histogramag = cv.calcHist([canal_verde], [0], None, [256], [0, 256])
# plt.plot(histogramar)
# plt.plot(histogramag)
# plt.plot(histogramab)
# plt.title('2Histograma de la imagen')
# plt.show()
# print(starry_img.shape)

# 4. Equalize image

equalize_head = cv.equalizeHist(mri_head_grayColor)
eq_horizontal = np.hstack((mri_head_grayColor, equalize_head))
cv.imshow('EqualizeHist Mri head Image', eq_horizontal)
k = cv.waitKey(0)
eq_vertical = np.vstack((mri_head_grayColor, equalize_head))
cv.imshow('EqualizeHist Mri head Image', eq_vertical)
k = cv.waitKey(0)

# 5. Local equalization

hidden_img = cv.imread('../imagenes/hidden-symbols.tif', cv.IMREAD_GRAYSCALE)
clahe = cv.createCLAHE(clipLimit=1, tileGridSize= [90,20])
imagen_ecualizada = clahe.apply(hidden_img)
eq_2_compara=np.hstack((hidden_img,imagen_ecualizada))
cv.imshow('EqualizeHIDDEN Mri head Image', eq_2_compara)
k = cv.waitKey(0)

# 6.  Contrast stretching

original = head_img.copy()
xp = [original.min(), original.max()]
fp = [100,200]
x = np.arange(256)
table = np.interp(x, xp, fp).astype('uint8')
wash_head = cv.LUT(head_img,table)
cv.imshow("original", original)
cv.imshow("Washed head", wash_head)
cv.waitKey(0)
# cv.destroyAllWindows()
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#7. Using normalize para elevar contraste
norm_head = cv.normalize(wash_head,None,alpha=0,beta=1,norm_type=cv.NORM_MINMAX,dtype=cv.CV_32F)
#scale to uint8
norm_head = (255*norm_head).astype(np.uint8)
cv.imshow("Normalize:Washed head", norm_head)
cv.waitKey(0)


# 8. Normalizando exclusivamente con normalize()
# alpha: Valor mínimo después de la normalización (en este caso, 0 para el rango [0, 1]).
# beta: Valor máximo después de la normalización (en este caso, 1 para el rango [0, 1]).
# ret,thresh = cv.threshold(wash_head,110,255,cv.THRESH_BINARY) 
norm_head = cv.normalize(wash_head,None,0,255,norm_type=cv.NORM_MINMAX,dtype=cv.CV_8U)
cv.imshow("usando Threshold y Normalize:Washed head", norm_head)
cv.waitKey(0)

# First, scale the range of the image to [0, 1] using norm= pix-minpix/maxpix-minpix(es el rango)
normalized_image = (wash_head - np.min(wash_head)) / (np.max(wash_head) - np.min(wash_head))
# Then, scale the dynamic gray level range to [0, 255] and convert to uint8
contrast_stretched_image = (normalized_image * 255).astype(np.uint8)
cv.imshow("Formula:Washed Image Normalizada", contrast_stretched_image)
cv.waitKey(0)

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////
# 9. Spatial convolution: Average filter
starry_img_bw = cv.imread('../imagenes/starry_night.jpg',cv.IMREAD_GRAYSCALE)
kernel_size = np.ones((5,5),np.float32)/25
# blurred_image = cv.blur(starry_img_bw, kernel_size)
average_starry = cv.filter2D(starry_img_bw,-1,kernel_size)
cv.imshow("Filter2D:Starry Night", average_starry)
cv.waitKey(0)

# 10. Gaussian blurring
blur_Gaus = cv.GaussianBlur(starry_img,(5,5),0)
cv.imshow("GaussianBlur:Starry Night", average_starry)
cv.waitKey(0)

# 11. Laplacian
sudoku_bw = cv.imread('../imagenes/sudoku.png',cv.IMREAD_GRAYSCALE)
cv.imshow("Sudoku", sudoku_bw)
sudoku_Gaus = cv.GaussianBlur(sudoku_bw,(5,5),0)
cv.imshow("Gaus:Sudoku", sudoku_Gaus)
laplacian_sudoku = cv.Laplacian(sudoku_Gaus,cv.CV_64F)
cv.imshow("Laplacian:Sudoku", laplacian_sudoku)
cv.waitKey(0)

# 12. Use convertScaleAbs() operation from OpenCV to bring it to positive values
scaleAbs_sudoku=cv.convertScaleAbs(sudoku_bw,alpha=0,beta=1)
cv.imshow("ConvertScaleAbs:Sudoku", scaleAbs_sudoku)
normalize_sudoku = cv.normalize(laplacian_sudoku,None,alpha=0,beta=1,norm_type=cv.NORM_MINMAX,dtype=cv.CV_32F)
# cv.imshow("Normalize:Sudoku", normalize_sudoku)
union_sudoku=np.hstack((scaleAbs_sudoku,normalize_sudoku))
cv.imshow("Normalize:Convertscale:Sudoku", union_sudoku)
cv.waitKey(0)

# 13. Sobel
sobelx = cv.Sobel(sudoku_bw,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(sudoku_bw,cv.CV_64F,0,1,ksize=5)
plt.subplot(2,2,1),plt.imshow(sudoku_bw,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian_sudoku,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
magnitud_sobel=cv.magnitude(sobelx,sobely)
magnitud_normalized = cv.normalize(magnitud_sobel, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
cv.imshow("Magnitud:Sobel", magnitud_normalized)
cv.waitKey(0)
plt.show()

# Sobel con Gaussiano
sobelx = cv.Sobel(sudoku_Gaus,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(sudoku_Gaus,cv.CV_64F,0,1,ksize=5)
plt.subplot(2,2,1),plt.imshow(sudoku_bw,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian_sudoku,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()

# 14. Median

circuit_img = cv.imread('../imagenes/circuit.jpg',cv.IMREAD_GRAYSCALE)
circuit_Gaus = cv.GaussianBlur(circuit_img,(5,5),0)
cv.imshow("Gaus:Circuit", circuit_Gaus)

median_circuit = cv.medianBlur(circuit_img,5)
cv.imshow("Median:Circuit", median_circuit)
cv.waitKey(0)

# 15. Discrete Fourier Transform (DFT)
dft = cv.dft(np.float32(circuit_img),flags = cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
plt.subplot(121),plt.imshow(circuit_img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# rows, cols = circuit_img.shape
# crow,ccol = rows/2 , cols/2
# # create a mask first, center square is 1, remaining all zeros
# mask = np.zeros((rows,cols,2),np.uint8)
# mask[crow-30:crow+30, ccol-30:ccol+30] = 1
# # apply mask and inverse DFT
# fshift = dft_shift*mask
# f_ishift = np.fft.ifftshift(fshift)
# img_back = cv.idft(f_ishift)
# img_back = cv.magnitude(img_back[:,:,0],img_back[:,:,1])
# plt.subplot(121),plt.imshow(circuit_img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# 16. fft.fftshift()

orig_spect_odd= np.array ([[0,10,20,90,100], [30,40,50,110,120], [60,70,80,130,140], [150,160,170,210,220], [180,190,200,230,240]], dtype=np.uint8)
orig_spect_even=np.array([[0,7,14,63,70,77], [21,28,35,84,91,98],[42,49,56,105,112,119], [126,133,140,189,196,203],[147,154,161,210,217,224],[168,175,182,231,238,245]],dtype=np.uint8)
original_Cuadrados = plt.imshow(orig_spect_odd, interpolation='nearest')
plt.show()
print("El tamaño de orig_spect_odd",orig_spect_odd.size)
print("El tamaño de orig_spect_even",orig_spect_even.shape)
# Naming a window 
cv.namedWindow("Resized_Window", cv.WINDOW_NORMAL) 
cv.resizeWindow("Resized_Window", 500, 500) 
cv.imshow("Resized_Window", orig_spect_odd) 
cv.waitKey(0) 

spectrum = np.fft.fft(orig_spect_odd)
magnitude = np.abs(spectrum)  # Magnitude of the spectrum
phase = np.angle(spectrum) 
spectrum = np.fft.fft(orig_spect_odd)

# Acceder a la parte real e imaginaria de los elementos del espectro
real_part = spectrum.real
imaginary_part = spectrum.imag
mag, phase = cv.cartToPolar(real_part, imaginary_part)


dft_cuadrado = cv.dft(np.float32(orig_spect_odd),flags = cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft_cuadrado)
magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
cv.imshow("Odd", magnitude_spectrum) 
cv.waitKey(0) 