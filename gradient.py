import numpy as py

x_old = 0
x_new = 6
gamma = 0.01

precision = 0.00001

def df(x):
    y = 4 * x ** 3 - 9 * x ** 2
    return y

while abs(x_new - x_old) > precision:
    x_old = x_new
    x_new += -gamma * df(x_old)

print("The local minimum occurs at %f" % x_new)
