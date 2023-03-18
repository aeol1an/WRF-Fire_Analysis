
import numpy as np
import mayavi.mlab as mlab
from tvtk.util.ctf import ColorTransferFunction, PiecewiseFunction
from matplotlib.pyplot import cm
from math import exp

def _cmap_to_ctf(data, cmap_name):
    values = list(np.linspace(np.nanmin(data), np.nanmax(data), 256))
    colors = list(np.linspace(0, 1, 256))
    cmap = cm.get_cmap(cmap_name)(colors)
    transfer_function = ColorTransferFunction()
    for i in range(255, -1, -1):
        transfer_function.add_rgb_point(values[i], cmap[i, 0], cmap[i, 1], cmap[i, 2])
    return transfer_function

def _modify_opacity(data, Squish, Disp):
    values = list(np.linspace(np.nanmin(data), np.nanmax(data), 256))
    opacities = list(np.linspace(0, 1, 256))
    otf = PiecewiseFunction()
    for i in range(255, -1, -1):
        otf.add_point(values[i], 1/(1+exp((-Squish)*(opacities[i]-Disp))))
    return otf

def plot3d(x3, y3, z3, vol_scalar, x2, y2, z2, surf_scalar,
           view, frame_num, num_frames, squish=15, disp=0.5, 
           out_filename = None,
           mpl_vol_cmap = "YlOrBr", mv_ter_cmap = "RdYlGn", r_mv_cmap = False):
    # Plot scatter with mayavi
    figure = mlab.figure('DensityPlot', size =(1920, 1080))
    try:
        grid = mlab.pipeline.scalar_field(x3, y3, z3, vol_scalar)
        vol = mlab.pipeline.volume(grid)
        ctf = _cmap_to_ctf(vol_scalar, mpl_vol_cmap)
        vol._volume_property.set_color(ctf)
        vol._ctf = ctf
        vol.update_ctf = True
        otf = _modify_opacity(vol_scalar, Squish=squish, Disp=disp)
        vol._otf = otf
        vol._volume_property.set_scalar_opacity(otf)

        # Create the data source
        src = mlab.pipeline.array2d_source(x2, y2, z2)

        dataset = src.mlab_source.dataset
        array_id = dataset.point_data.add_array(surf_scalar.T.ravel())
        dataset.point_data.get_array(array_id).name = 'color'
        dataset.point_data.update()

        # Here, we build the very exact pipeline of surf, but add a
        # set_active_attribute filter to switch the color, this is code very
        # similar to the code introduced in:
        # http://code.enthought.com/projects/mayavi/docs/development/html/mayavi/mlab.html#assembling-pipelines-with-mlab
        warp = mlab.pipeline.warp_scalar(src, warp_scale=.5)
        normals = mlab.pipeline.poly_data_normals(warp)
        active_attr = mlab.pipeline.set_active_attribute(normals,
                                                    point_scalars='color')
        surf = mlab.pipeline.surface(active_attr, colormap=mv_ter_cmap)
        surf.module_manager.scalar_lut_manager.reverse_lut = r_mv_cmap

        a, e, d, f = view(frame_num, num_frames)
        mlab.view(azimuth=a, elevation=e, distance=d, focalpoint=f)

        if not(out_filename is None):
            mlab.savefig(out_filename)
    except Exception:
        mlab.close(figure)
        raise
    return figure