from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

def fun(x1,x2):
    return 6 + np.power(x1,2) + np.power(x2,2)

fig = plt.figure()
axis = plt.axes(projection= '3d')
x1 = np.arange(-5,5,0.1)
x2 = np.arange(-5,5,0.1)
X, Y = np.meshgrid(x1,x2)
Z = fun(X,Y)
axis.contourf(X, Y, Z,200,cmap='CMRmap')
axis.set_xlabel('x1')
axis.set_ylabel('x2')
axis.set_zlabel('y1(x)')
plt.title('Plotting the function y1(x)')
plt.show()

fig1 = plt.figure()
axis = plt.axes(projection='3d')
y2=8
y2 = np.asarray(y2)
axis.plot3D(X,Y,y2,'gray')
plt.title('Plotting the function y2(x)')
plt.show()


#Decision boundary is a circle
c = plt.Circle((0,0),1,color='r')
fig, ax = plt.subplots()
ax.set_xlim((-2,2))
ax.set_ylim((-2,2))
ax.add_patch(c)
plt.show()

