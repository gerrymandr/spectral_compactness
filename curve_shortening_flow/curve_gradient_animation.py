import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import curve_gradient_descent as gd 
import shapely.geometry as geom
import curve_utils
import random
import geopandas as gpd

# First set up the figure, the axis, and the plot element we want to animate
# fig = plt.figure(figsize=(6,6))
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
# ax1 = plt.axes(xlim=(-4, 4), ylim=(-4, 4))
ax1.set_xlim(-108,-92)
ax1.set_ylim(24,38)
ax2.set_xlim(0,5000)
ax2.set_ylim(0,1)
ax2.set_xlabel("Steps in flow")
ax2.set_ylabel("Polsby-Popper score")
line1, = ax1.plot([], [])
line2, = ax2.plot([], [])

pp_score = ax1.text(0.02, .9, '', transform=ax1.transAxes)
# step_num = ax1.text(0.02, .5, '', transform=ax1.transAxes)
plt.plot([], [])

# Set up the curve whose flow we will animate
# curve = np.array([[0,0],[1,1],[1,2],[0,3],[-1,2],[-1,1]])
texas = curve_utils.curve_from_shapefile('shapefiles\\Texas outline\\Texas_State_Boundary.shp', tolerance=0.03)
coords = texas.exterior.coords.xy
curve = np.array([[x,y] for (x,y) in zip(coords[0], coords[1])])
curve = curve_utils.subdivide_curve(curve_utils.subdivide_curve(curve, threshold=1), threshold=1)
# curve = np.array([[np.cos(theta), 5*np.sin(theta)] for theta in np.linspace(0, 2*np.pi, 50, endpoint=False)])
# curve = curve_utils.subdivide_curve(np.array([[0,-5],[1,-4],[2,-2],[4,-1],[5,0],[4,1],[2,2],[1,4],[0,5],[-1,4],[-2,2],[-4,1],[-5,0],[-4,-1],[-2,-2],[-1,-4]]))
# curve = curve_utils.subdivide_curve(np.array([[-3,-3],[3,-3],[3,3],[-3,3]]))
# curve = curve_utils.subdivide_curve(curve_utils.subdivide_curve(np.array([[0,0],[3,3],[3,5],[-2,0],[3,-5],[3,-3]])))
# curve = np.array([[4*np.cos(theta) + 2*random.random(), 4*np.sin(theta) + 2*random.random()] for theta in np.linspace(0, 2*np.pi, 75, endpoint=False)])
# curve = np.array([[x, 4*np.sin(np.pi*x/2)] for x in np.linspace(-4, 4, 30)])
# curve = np.append(curve, np.array([[x, 4*np.sin(np.pi*x/2) - 2] for x in np.linspace(4, -4, 30)]), axis=0)
pp_data = np.empty((0,2), float)
# pt = np.array([0.0,0.0])
# curve = np.array([pt])
# for i in range(250):
# 	pt += np.array([random.uniform(-0.5,0.5), random.uniform(-0.5,0.5)])
# 	curve = np.append(curve, [pt], axis=0)

# curve = np.array([[np.cos(theta), np.sin(theta) - 4] for theta in np.linspace(1.5 * np.pi, 2 * np.pi, 5, endpoint=False)])


# initialization function: plot the background of each frame
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    pp_score.set_text('')
    # step_num.set_text('')
    return [line1, line2]

c_len = 0

def animate(i):
    global curve
    global pp_data
    global c_len 
    if len(curve) <= 4:
    	return line1, line2, pp_score
    curve = gd.flow_step(curve)
    if len(curve) != c_len:
    	c_len = len(curve)
    	print "New length: {}".format(c_len)
    curve_x, curve_y = np.append(curve[:,0], curve[0,0]), np.append(curve[:,1], curve[0,1])
    line1.set_data(curve_x, curve_y)
    pp = polsby_popper(np.append(curve, np.array([curve[0]]), axis=0))
    pp_data = np.append(pp_data, np.array([[i, pp]]), axis=0)
    line2.set_data(pp_data[:,0], pp_data[:,1])
    pp_score.set_text("Step #: {}\nPolsby-Popper score:\n {}".format(i, pp))
    # step_num.set_text("Step #:\n {}".format(i))
    return line1, line2, pp_score

def polsby_popper(closed_curve): 
	region = geom.Polygon(closed_curve)
	return (4 * np.pi * region.area) / (region.length ** 2)

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=5000, interval=1, blit=True)

Writer = animation.writers['ffmpeg']
ffmpeg = Writer(fps=300, metadata=dict(artist='Me'), bitrate=1800)
anim.save('global_curves.mp4', writer=ffmpeg)
# anim.save('curve_animation.html', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()