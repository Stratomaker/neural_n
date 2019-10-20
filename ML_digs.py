from sklearn import datasets
from sklearn.svm import SVC
import scipy as sc


digits = datasets.load_digits()
features = digits.data 
labels = digits.target

clf = SVC(gamma = 0.001)
clf.fit(features, labels)


img = sc.misc.imread("FSP2.jpg")
img = sc.misc.imresize(img, (8,8))
img = img.astype(digits.images.dtype)
print(img.dtype)
img = sc.misc.bytescale(img, high=16, low=0)


x_test = []

for eachRow in img:
	for eachPixel in eachRow:
		x_test.append(sum(eachPixel)/3.0)



print(clf.predict([x_test]))



'''import matplotlib.pyplot as plt 

from sklearn import datasets
from sklearn import svm
from scipy import misc 
import numpy as np

digits = datasets.load_digits()

clf = svm.SVC(gamma = 0.0001, C=100)

print(len(digits.data))

x,y = digits.data[:-10], digits.target[:-10]
clf.fit(x,y)

print('Prediction: ', clf.predict([digits.data[-1]]))
plt.imshow(digits.images[-1], cmap = plt.cm.gray_r, interpolation = "nearest")
plt.show()

img = misc.imread('FSP2.jpg')
img = misc.imresize(img, (8,8))
print(img)'''