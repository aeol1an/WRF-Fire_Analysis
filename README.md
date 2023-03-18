# Wrfout Idealized Plotting
## Getting the data
### HRRR sounding data
For the HRRR soundings you need to have access to a machine with ncl. Then, you can wget the grib2 from the download link specified in the conversion script then use the script to convert the grib2 to netCDF. To do this, change string on line 5 of sounding_grib2_to_netcdf.ncl to the name of your grib2 file and change out.nc on line 11 to your prefered output. If you would like to replicate my cases, my two hrrr grib2 files are in /glade/u/home/asnnaik/ATM255/final_proj/hrrr/. The creek fire case is "hrrr.t21z.wrfprsf00.grib2" and the loyalton fire case is "hrrr.t22z.wrfprsf00.grib2".

### Wrfout files
See the getwrfout.sh files in "./(tor|creek|loyalton)/nc/" replace the section in {} with your cheyenne login and run the script. This will get my wrfout files.

## Setting Up The Python Env
See "./create_conda_env.sh". These commands in your anaconda terminal should download all the libraries needed to run the jupyter notebook files.

## Creating the 3d plots
See "./gen_3d_smoke_plots.ipynb". Comment out the cases you don't want to render in the second cell. This notebook uses the mayaviplot.py file in which I coded a function to create a 2d surface and 3d volume plot and change the colormaps. The 2d surface is limited to [mayavi](https://docs.enthought.com/mayavi/mayavi/mlab_changing_object_looks.html) colormaps (variable mv_ter_cmap), while the 3d volume needs a [matplotlib](https://matplotlib.org/stable/tutorials/colors/colormaps.html) colormap (value mv_ter_cmap). If you would like to replace the 2d surface colormap, set r_mv_cmap to True.

You can fiddle with the "squish" and "displacement" variables to modify the opacity of the volume. Lower displacement (min 0) means lower values are more opaque, and higher displacement (max 1) means only the highest values will be visible. Squish will affect the opacity gradient, meaning that very high values (unlimited) of squish will look like solid objects, while lower values (min 0 i think) will look like clouds. If you are familar, squish and displacement modify the sigmoid function. 

## Creating the soundings and vorticity plots
For soundings, the main code is in sounding.py. A lot of it is adopted from the sharppy website. I just added parsers for HRRR data and input_sounding data to modify HRRR data then plot the sounding. The sounding class can import hrrr data through the constructor, accept modifications to the data through a given input_sounding file (this may have bugs, so separate your values with one space only), and output to a plot or another input_sounding for use in a wrf run.

I've only implemented k-curl and j-curl so far because that's all I needed. These are implemented in vort.py. I'm sure there's a cleaner way to implement them, but it was like 4AM when I wrote them, and I also know for a fact that the k-curl function is broken for a non-square domain. I plan to fix it in a few weeks. For my data though, since it is square, its okay, and should work.

These files are used to create plots in ./{case}/sounding_mpl.ipynb.