import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.widgets  import RectangleSelector
from PIL import Image

minx = 0 
miny = 0 
maxx = 0 
maxy = 0 

def line_select_callback(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    global minx
    global miny
    global maxy
    global maxy
    minx = min(x1,x2)
    miny = min(y1,y2)
    maxx = max(x1,x2)
    maxy = max(y1,y2)

def bbox_coordinates(img_path):
    dpi = 80.0
    image = Image.open(img_path).convert('RGB')
    figsize = (image.size[0]/dpi, image.size[1]/dpi)
    fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    im = ax.imshow(image, aspect='normal')
    rs = RectangleSelector(ax, line_select_callback,
                           drawtype='box', useblit=False, button=[1], 
                           minspanx=5, minspany=5, spancoords='pixels', 
                           interactive=True)

    plt.show()
    return np.array([minx, miny, maxx, maxy])   
