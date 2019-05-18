# Get a feeling of the eigenvalue calculation algorithm
# det(A-l*I) = 0
# for different lambda
# Code plots the linear transformation of [A - lambda * I] of the unit circle
# and the unit girds for different lambda

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pylab import *

################################################################################
# Read me:
# If you want to play for fun, alter the "A", "min_lambda", and "min_lambda"
# the animation will display the linear transformation of [A - lambda * I]
# Code only works for 2x2 matrix
A = np.array([[2, 1.5], [0.4, 3]]) # The matrix to be analyzed
min_lambda = 0
max_lambda = 5
################################################################################

x_max = 8 # where does the grid go on +/-x directions
y_max = 5 # where does the grid go on +/-y directions
gx = x_max * 2 - 1 # number of vertical grid lines
gy = y_max * 2 - 1 # number of horizontal grid lines

# locate the points on the outside box, used for grid plot
l = np.zeros((2, gy))
B_l = l.copy()
r = l.copy()
B_r = r.copy()
t = np.zeros((2, gx))
B_t = t.copy()
b = t.copy()
B_b = b.copy()

l[1, :] = np.linspace(-y_max+1, y_max-1, gy)
l[0, :] = l[1, :] * 0 - x_max
r[1, :] = np.linspace(-y_max+1, y_max-1, gy)
r[0, :] = r[1, :] * 0 + x_max
t[0, :] = np.linspace(-x_max+1, x_max-1, gx)
t[1, :] = t[0, :] * 0 + y_max
b[0, :] = np.linspace(-x_max+1, x_max-1, gx)
b[1, :] = t[0, :] * 0 - y_max

B = A.copy() # placeholder for the matrix: A - Lambda * I
th = np.arange(0.0, 2.1*np.pi, 0.1) # angle of the unit circle
c = np.array([np.cos(th), np.sin(th)]) # circle vector to be transformed
B_c = c.copy() # placeholder for vector: (A - Lambda * I) * unit circle

e, v = np.linalg.eig(A) # calculate eigven values (e) and eigen vectors (v)

fig, ax = plt.subplots()
fig.set_figheight(6)
fig.set_figwidth(9)
plt.plot(c[0, :], c[1, :], 'r-') # plot original unit circle
thismanager = get_current_fig_manager()
thismanager.window.wm_geometry("+500+0") # move figure window position
xdata, ydata = [], []
ln, = plt.plot([], [], color='xkcd:sky blue', linestyle='-', linewidth=0.5)
g, = [plt.plot([], [], color='xkcd:sky blue', linestyle='-', linewidth=1)]
for i in range(0, gx+gy):    
    if i < gx - 1:
        if i < (gx - 1) / 2 - 1:
            lw_temp = 0.5 # vertical grid width whoes original x < 0
        else:
            lw_temp = 2.5 # vertical grid width whoes original x >= 0
        g_temp, = plt.plot([], [], color='xkcd:sky blue',
                       linewidth=lw_temp, linestyle='-')
    else:
        if i - gx < ((gy +-1) / 2 - 1):
            lw_temp = 0.5 # horizontal grid width whoes original y < 0
        else:
            lw_temp = 2.5 # horizontal grid width whoes original y >= 0
        g_temp, = plt.plot([], [], color=(0.2, 0.8, 0.2),
                       linewidth=lw_temp, linestyle='-') # plot placeholder
    g.append(g_temp) # append list

# graph properties
ax.axis('equal') # make the plot aspect ratio = 1
ax.set_facecolor('k') # background color = black
plt.xlabel('Linear Transform of $A - \lambda  I$')
plt.title('Grids: unit grid after transformation of $A - \lambda  I$\n' +\
          'Red circle: unit circle $u$ before transformation\n' +\
          'Blue ellipse: unit circle after transformation: $(A - \lambda  I)u$')
tbox_1_t = '' # lambda value update text placeholder
tbox_1 = plt.text(7, -5, tbox_1_t,
                  {'color': 'xkcd:sky blue', 'fontsize': 18},
         va="top", ha="right")
tbox_1.set_bbox(dict(facecolor='grey', alpha=0.8, edgecolor='grey'))
tbox_2_t = 'eigenvalues of A: ' +\
    np.array2string(e, precision=2, separator=' ', suppress_small=True)
tbox_2 = plt.text(11.5, 7, tbox_2_t,
                  {'color': 'xkcd:sky blue', 'fontsize': 16},
         va="top", ha="right")
tbox_2.set_bbox(dict(facecolor='grey', alpha=0.8, edgecolor='grey'))
tbox_2_t = 'eigenvectors of A: \n' +\
    np.array2string(v, precision=2, separator=' ', suppress_small=True)
tbox_3 = plt.text(11.5, 5.5, tbox_2_t,
                  {'color': 'xkcd:sky blue', 'fontsize': 16},
         va="top", ha="right")
tbox_3.set_bbox(dict(facecolor='grey', alpha=0.8, edgecolor='grey'))
tbox_4_t = 'A = ' +\
    np.array2string(A, precision=2, separator=' ', suppress_small=True)
tbox_4 = plt.text(-1, -5, tbox_4_t,
                  {'color': 'xkcd:sky blue', 'fontsize': 18},
         va="top", ha="right")
tbox_4.set_bbox(dict(facecolor='grey', alpha=0.8, edgecolor='grey'))
             
# to create the animation
def init(): # initialize the animation
    plot_max = 12
    ax.set_xlim(-plot_max, plot_max)
    ax.set_ylim(-plot_max, plot_max)
    return ln,

def update(frame): # animation update function
    B = A - frame * np.identity(2) # calculate A - Lambda * I
    for i in range(len(th)):
        B_c[:, i] = np.dot(B, c[:, i]) # linear transform of unit circle
    for i in range(0, gy):
        B_l[:, i] = np.dot(B, l[:, i]) # linear transform of left box line
    for i in range(0, gy):
        B_r[:, i] = np.dot(B, r[:, i]) # linear transform of right box line
    for i in range(0, gx):
        B_t[:, i] = np.dot(B, t[:, i]) # linear transform of top box line
    for i in range(0, gx):
        B_b[:, i] = np.dot(B, b[:, i]) # linear transform of bottom box line

    xdata.append(B_c[0, :])
    ydata.append(B_c[1, :])
    ln.set_data(xdata, ydata)
    yield ln
    
    for i in range(0, gx):
        g[i].set_data([B_t[0, i], B_b[0, i]],
                      [B_t[1, i], B_b[1, i]])
        yield g[i] # return g[i]
    for i in range(gx, gx+gy):
        g[i].set_data([B_l[0, i-gx], B_r[0, i-gx]],
                      [B_l[1, i-gx], B_r[1, i-gx]])
        yield g[i] # return g[i]
        
    tbox_1.set_text('$\lambda$ =%.1f' % frame) # update lambda on animation
    yield tbox_1
    
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(min_lambda,max_lambda,500),
                   init_func=init, blit=True, repeat=True, interval=10)
plt.show()
ani.save('animation_1_1_1_1.gif', writer='imagemagick', fps=30)
