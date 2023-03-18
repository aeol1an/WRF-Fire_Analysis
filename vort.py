import numpy as np
from netCDF4 import Dataset
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from wrf import interplevel, to_np, getvar

def _discrete_curl(a0, a1, b0, b1, da, db):
    return ((b1-b0)/da)-((a1-a0)/db)

def _mean_interp(x0, x1, y0, y1):
    return np.mean([x0, x1, y0, y1])

def kvort(U_wrf, V_wrf, level):
    u = U_wrf.transpose(2, 1, 0)[:,:,level]
    v = V_wrf.transpose(2, 1, 0)[:,:,level]

    _,usize = u.shape
    vsize,_ = v.shape
    u0 = u[1:usize,0:usize-1]
    u1 = u[1:usize,1:usize]
    v0 = v[0:vsize-1,1:vsize]
    v1 = v[1:vsize,1:vsize]
    curl = np.vectorize(_discrete_curl, excluded=['da', 'db'])
    interp = np.vectorize(_mean_interp)
    rawvort = curl(u0, u1, v0, v1, 50, 50)
    num_x, num_y = rawvort.shape
    rawvortx1y1 = rawvort[0:num_x-1,0:num_y-1]
    rawvortx1y2 = rawvort[0:num_x-1,1:num_y]
    rawvortx2y1 = rawvort[1:num_x,0:num_y-1]
    rawvortx2y2 = rawvort[1:num_x,1:num_y]
    a = np.zeros((usize,vsize))
    a[1:usize-1, 1:vsize-1] = interp(rawvortx1y1, rawvortx1y2, rawvortx2y1, rawvortx2y2)
    return a

def plotkvortw(frame, file, bounds, datapath, targetdir, level):
    nc = Dataset(datapath/file)
    U = nc["U"][0,:,:,:]
    V = nc["V"][0,:,:,:]
    W = nc["W"][0,:,:,:]

    x1, x2, y1, y2 = bounds

    aspect_ratio = 16*(x2-x1)/(y2-y1)

    avo = kvort(U, V, level)[x1:x2,y1:y2]
    w = W.transpose(2, 1, 0)[x1:x2,y1:y2,level]

    xm, ym = 50*np.mgrid[x1:x2, y1:y2]
    
    fig = plt.figure(figsize=(aspect_ratio, 12))
    ax = plt.axes()

    if not(np.all(w==0)):
        cont = plt.contour(xm, ym, w, [0, 5, 15], cmap=plt.get_cmap("rainbow"))
        ax.clabel(cont, [0, 5, 15], inline=True)

    contf = plt.contourf(xm, ym, avo, np.linspace(-0.2, 0.2, 33), cmap=plt.get_cmap("bwr").with_extremes(over='darkred', under='darkblue'), extend='both')
    cb = fig.colorbar(contf, orientation='horizontal', fraction=0.2, pad=0.075, shrink=0.8, ticks=np.linspace(-0.2, 0.2, 9), extend='both')
    cb.set_label(label='Vorticity (s$^{-1}$)', size=15, weight='bold')
    cb.ax.set_xticklabels(np.linspace(-0.2, 0.2, 9))
    cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.set_aspect('equal')

    plt.title("K Component Vorticity (s$^{-1}$, shaded) and Vertical Velocity ($\\frac{m}{s}$, contour lines) ", pad=15, size=15, weight='bold')
    plt.xlabel("x coordinate (m)", size=10)
    plt.ylabel("y coordinate (m)", size=10)
    plt.savefig(targetdir/(str(frame)+".png"), dpi=fig.dpi, bbox_inches='tight')
    plt.close(fig)

def jvort(U_wrf, W_wrf, level):
    u = U_wrf.transpose(2, 1, 0)[:,level,:]
    w = W_wrf.transpose(2, 1, 0)[:,level,:]

    usize,_ = u.shape
    usize-=1
    _,wsize = w.shape
    wsize-=1
    u0 = u[0:usize,0:wsize]
    u1 = u[0:usize,1:wsize+1]
    w0 = w[0:usize,0:wsize]
    w1 = w[1:usize+1,0:wsize]
    curl = np.vectorize(_discrete_curl, excluded=['da', 'db'])
    interp = np.vectorize(_mean_interp)
    rawvort = curl(w0, w1, u0, u1, 50, 50)
    num_x, num_y = rawvort.shape
    rawvortx1y1 = rawvort[0:num_x-1,0:num_y-1]
    rawvortx1y2 = rawvort[0:num_x-1,1:num_y]
    rawvortx2y1 = rawvort[1:num_x,0:num_y-1]
    rawvortx2y2 = rawvort[1:num_x,1:num_y]
    a = np.zeros((usize+1,wsize+1))
    a[1:usize, 1:wsize] = interp(rawvortx1y1, rawvortx1y2, rawvortx2y1, rawvortx2y2)
    return a

def plotjvort(frame, file, datapath, targetdir, level):
    nc = Dataset(datapath/file)
    hgt = to_np(getvar(nc, 'height_agl', units='m'))
    U = nc["U"][0,:,:,:]
    _, _, usize = U.shape
    U = U[:,:,1:usize]
    W = nc["W"][0,:,:,:]
    wsize, _, _ = W.shape
    W = W[1:wsize,:,:]

    Uinterp = interplevel(U, hgt, 150)
    for h in range(200,4000,50):
        Uinterp = np.dstack((Uinterp, interplevel(U, hgt, h)))
    Winterp = interplevel(W, hgt, 150)
    for h in range(200,4000,50):
        Winterp = np.dstack((Winterp, interplevel(W, hgt, h)))
    Uinterp = Uinterp.transpose(2, 0, 1)
    Winterp = Winterp.transpose(2, 0, 1)

    avo = jvort(Uinterp, Winterp, level)
    x, y = avo.shape

    xm, ym = 50*np.mgrid[0:x, 0:y]
    
    fig = plt.figure(figsize=(round(x*12/y), 12))
    ax = plt.axes()


    contf = plt.contourf(xm, ym, avo, np.linspace(-0.2, 0.2, 33), cmap=plt.get_cmap("bwr").with_extremes(over='darkred', under='darkblue'), extend='both')
    cb = fig.colorbar(contf, orientation='horizontal', fraction=0.2, pad=0.075, shrink=0.8, ticks=np.linspace(-0.2, 0.2, 9), extend='both')
    cb.set_label(label='Vorticity (s$^{-1}$)', size=15, weight='bold')
    cb.ax.set_xticklabels(np.linspace(-0.2, 0.2, 9))
    cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.set_aspect('equal')

    plt.title("J Component Vorticity (s$^{-1}$, shaded)", pad=15, size=15, weight='bold')
    plt.xlabel("x coordinate (m)", size=10)
    plt.ylabel("z coordinate (m)", size=10)
    plt.savefig(targetdir/(str(frame)+".png"), dpi=fig.dpi, bbox_inches='tight')
    plt.close(fig)

def getnumj(file, datapath):
    nc = Dataset(datapath/file)
    U = nc["U"][0,:,:,:]
    _, y, _ = U.shape
    return y