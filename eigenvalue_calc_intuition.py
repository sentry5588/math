# Get a feeling of the matrix addition
# by using the eigenvalue calculation algorithm
# det(A-l*x) = 0

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

th = np.arange(0.0, 2.0*np.pi, 0.1)
c = np.array([np.cos(th), np.sin(th)])

A = np.array([[2.0, 1.0], [0.0, 4.0]])
A_lambdaI = A.copy()

Ac = c.copy()
A_lambdaI_c = c.copy()


fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'b')


def init():
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    return ln,

def update(frame):
    A_lambdaI = A - frame * np.array([[1.0, 0.0], [0.0, 1.0]])
    for i in range(len(th)):
        A_lambdaI_c[:, i] = np.dot(A_lambdaI, c[:, i])
    xdata.append(A_lambdaI_c[0, :])
    ydata.append(A_lambdaI_c[1, :])
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 5.0, 300), 
                    init_func=init, blit=True, repeat=False, interval=10)
plt.show()

