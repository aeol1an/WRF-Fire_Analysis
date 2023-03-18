conda create -n wrf python=3.8
conda activate wrf

conda install netcdf4=1.5.7
conda install matplotlib=3.6.2
conda install cartopy=0.21.1
conda install jupyter=1.0.0
conda install -c conda-forge wrf-python=1.3.4.1
conda install -c conda-forge sharppy

pip install pint==0.20.1
pip install metpy==1.3.0
pip install mayavi
pip install ipywidgets
pip install ipyevents

jupyter nbextension install --py mayavi --user
jupyter nbextension enable --py mayavi --user