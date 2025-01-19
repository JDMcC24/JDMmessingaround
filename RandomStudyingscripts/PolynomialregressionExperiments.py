import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import time


""" Linear Data"""

# #Generating Data from normal distribution with normal errors
# n=1000 #Number of data points
# xs = np.random.randn(n,1)*2 +2
# errors = np.random.randn(n,1) * 1.5
# b = 1.5
# c = 1
# ys = c + b * xs +  errors

# #Plotting data
# plt.scatter(xs,ys, label = 'Data')
# plt.xlim(0, 1.1*max(xs))
# plt.ylim(0, 1.1*max(ys))

# starttime = time.time()
# #Best Fit Regression Line using Normal Equations
# A = np.ones(np.shape(xs))
# A = np.concatenate((A,xs), axis = 1)
# coeff = np.linalg.inv(A.T@ A)@ A.T @ ys
# dom = np.linspace(0,1.1*max(xs))
# regline = coeff[0]+ coeff[1]*dom
# plt.plot(dom, regline, color = 'red', label = 'Best Fit line')
# regtime = time.time() - starttime
# starttime = time.time()

# #Modeling with Sklearn ML
# model = LinearRegression()
# model.fit(xs,ys)
# modelline = model.predict(dom)
# plt.plot(dom, modelline, color = 'green', label = 'ML-Line')
# plt.legend(loc = 'upper left')
# modeltime = time.time() - starttime
# print(f'Normal Equations Time: {regtime} seconds. ML Time: {modeltime} seconds.')
# plt.show()




"""Polynomial Data"""
n= 100 #Number of data points
d = 2 #Degree of polynomial Must be at least 2
xs = np.random.randn(n,1) +2
dom = np.linspace(1.1* min(xs),1.1*max(xs),n)
ys = np.zeros(np.shape(xs))
eq=" y = "
for i in range(d+1):
    coeff = (np.random.randn(1))**(i+1)
    ys+= coeff*xs**i
    eq+= f'{round(coeff[0],2)}x^{i} +'
erroraverage = np.abs(np.random.randn(1))
ys += np.random.randn(n,1) * erroraverage 
eq = eq[:-1]
eq+= f'average error is {round(erroraverage[0],2)}'

#Plotting data
plt.scatter(xs,ys, label = 'Data'+ eq, s = 10)
plt.xlim(-.5, 1.1*max(xs))
plt.ylim(-.5, 1.1*max(ys))


#Poly Regression

starttime = time.time()
A = np.ones(np.shape(xs))
for i in range(1,d+1):
    A = np.concatenate((A,xs**i), axis = 1)

coeff = np.linalg.inv(A.T@ A)@ A.T @ ys
regline = coeff[0]* np.ones(np.shape(dom))
for i in range(1,d+1):
    regline += coeff[i]*dom**i
plt.plot(dom, regline, color = 'red', label = f'Best Fit Curve: {round(coeff[0][0],2)} + {round(coeff[1][0],2)}x + {round(coeff[2][0],2)}x^2')

regtime = time.time() - starttime


#Modeling with Sklearn ML

starttime = time.time() 
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = d)
model = LinearRegression()
xs_poly = poly.fit_transform(xs)
model.fit(xs_poly,ys)
xs_range_poly = poly.transform(dom)
ys_pred = model.predict(xs_range_poly)
plt.plot(dom, ys_pred, color = 'green', label = 'ML-Curve')
MLtime = time.time() - starttime    
plt.legend(loc = 'upper left')
print('Normal Equations Time: ', regtime, 'seconds. ML Time: ', MLtime, 'seconds.') 
plt.show()

"""Take aways:
Noraml equations were typically much faster (about a third of the time for basic linear regression)
and is by definition the most accurate. However, ML produced essentially the same curves, was easier to implement,
and avoids any potential issues of singularity of the matrices.
The more data points and with the higher degree of polynomials, ML begame relatively more efficient."""