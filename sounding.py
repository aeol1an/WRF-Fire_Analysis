import numpy as np
import pandas as pd
from netCDF4 import Dataset
from math import radians

import warnings
warnings.filterwarnings("ignore")
import sharppy.plot.skew as skew
from matplotlib.ticker import ScalarFormatter, MultipleLocator
from matplotlib.collections import LineCollection
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import gridspec
from sharppy.sharptab import winds, utils, params, thermo, interp, profile
from skewx import SkewXAxes
from matplotlib.projections import register_projection

register_projection(SkewXAxes)

def _haversine(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2, lat2 = map(radians, [lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.arcsin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))

def _fmt(value, fmt='int'):
    if fmt == 'int':
        try:
            val = int(value)
        except:
            val = str("M")
    else:
        try:
            val = round(value,1)
        except:
            val = "M"
    return val


class SoundingData:
    def __init__(self, type: str, filename: str, lat: float, lon: float):
        if (type == 'hrrr'):
            #Input
            self.lat = lat
            self.lon = lon
            hrrr_sounding = Dataset(filename)
            distances = _haversine((hrrr_sounding["gridlon_0"][:,:]), (hrrr_sounding["gridlat_0"][:,:]), lon, lat)
            row, col = np.unravel_index(distances.argmin(), distances.shape)

            #Raw
            self.ter = hrrr_sounding["HGT_P0_L1_GLC0"][row, col]
            col_vars = {
                "P": (list(range(50, 1001, 25)) + [1013.2])[::-1],
                "Z": np.flip(hrrr_sounding["HGT_P0_L100_GLC0"][:,row,col] - self.ter),
                "T": np.flip(hrrr_sounding["TMP_P0_L100_GLC0"][:,row,col]),
                "q": np.flip(hrrr_sounding["SPFH_P0_L100_GLC0"][:,row,col]),
                "Td": np.flip(hrrr_sounding["DPT_P0_L100_GLC0"][:,row,col]),
                "U": np.flip(hrrr_sounding["UGRD_P0_L100_GLC0"][:,row,col]),
                "V": np.flip(hrrr_sounding["VGRD_P0_L100_GLC0"][:,row,col]),
            }
            self.profile = pd.DataFrame(col_vars)
            self.profile = (self.profile[self.profile['Z'] >= 0]).reset_index()
            self.sfc_P = hrrr_sounding["PRES_P0_L1_GLC0"][row, col]/100
            self.sfc_Theta = hrrr_sounding["POT_P0_L103_GLC0"][row, col]
            self.sfc_q = hrrr_sounding["SPFH_P0_L103_GLC0"][row, col]
            self.lat = hrrr_sounding["gridlat_0"][row, col]
            self.lon = hrrr_sounding["gridlon_0"][row, col]

            #Derived
            self.profile["Theta"] = self.profile["T"]*(1000/self.profile["P"])**0.2854
            self.profile["w"] = self.profile["q"]/(1-self.profile["q"])
            self.profile["wspeed"] = np.hypot(self.profile["U"], self.profile["V"])
            self.profile["wspeedkt"] = np.hypot(self.profile["U"], self.profile["V"])*1.94384
            self.profile["wdir"] = np.abs((np.sign(self.profile["U"])-1)/2) * np.degrees(np.arccos(-self.profile["V"]/self.profile["wspeed"])) + \
                                   np.abs((np.sign(self.profile["U"])+1)/2) * (360-np.degrees(np.arccos(-self.profile["V"]/self.profile["wspeed"])))
            self.sfc_w = self.sfc_q/(1-self.sfc_q)
            self.sfc_q *= 1000
            self.sfc_w *= 1000
            self.profile["q"] *= 1000
            self.profile["w"] *= 1000
        else:
            raise TypeError("Invalid model type")
    
    def gen_wrf_sounding(self, filename: str):
        sounding_str = ""
        #sfc
        sounding_str += " " + str(self.sfc_P) + " " + str(self.sfc_Theta) + " " + str(self.sfc_w) + "\n"
        #profile
        for _, row in self.profile.iterrows():
            sounding_str += "  " + str(row["Z"]) + " " + str(row["Theta"]) + " " + str(row["w"]) + " " + str(row["U"]) + " " + str(row["V"]) + "\n"
        with open(filename, "w") as f:
            f.write(sounding_str)

    def modify_from_wrf(self, filename: str):
        with open(filename, "r") as f:
            wrf = f.read().split('\n')
            wrf = wrf[1:len(wrf)-1]
        for i in range(len(wrf)):
            wrf[i] = wrf[i].split()
        wrf = np.asarray(wrf, np.float32)
        U = wrf[:,3]
        V = wrf[:,4]
        length_needed = len(self.profile.index)
        U = np.append(U, np.zeros(length_needed-len(U))-9999.00)
        V = np.append(V, np.zeros(length_needed-len(V))-9999.00)
        self.profile["U"] = U
        self.profile["V"] = V
        self.profile["wspeed"] = np.hypot(self.profile["U"], self.profile["V"])
        self.profile["wspeedkt"] = np.hypot(self.profile["U"], self.profile["V"])*1.94384
        self.profile["wdir"] = np.abs((np.sign(self.profile["U"])-1)/2) * np.degrees(np.arccos(-self.profile["V"]/self.profile["wspeed"])) + \
                                np.abs((np.sign(self.profile["U"])+1)/2) * (360-np.degrees(np.arccos(-self.profile["V"]/self.profile["wspeed"])))
        for i in self.profile.index:
            if self.profile["U"][i] == -9999.00 or self.profile["V"][i] == -9999.00:
                self.profile["wspeed"][i] = -9999.00
                self.profile["wspeedkt"][i] = -9999.00
                self.profile["wdir"][i] = -9999.00


    def print_profile(self):
        print(self.profile)

    def plot(self, date, filename: str = None, title: str = ''):
        prof = profile.create_profile(pres=self.profile["P"], hght=(self.profile["Z"]+self.ter), tmpc=(self.profile["T"]-273.15), 
                                      dwpc=self.profile["Td"]-273.15, wspd=self.profile["wspeedkt"], wdir=self.profile["wdir"], missing=-9999.00,
                                      strictQC=False, profile='convective', date=date)
        # pb_plot=1050
        # pt_plot=100
        # dp_plot=10
        # plevs_plot = np.arange(pb_plot,pt_plot-1,-dp_plot)
        # Open up the text file with the data in columns (e.g. the sample OAX file distributed with SHARPpy)

        # Set up the figure in matplotlib.
        fig = plt.figure(figsize=(9, 8))
        gs = gridspec.GridSpec(4,4, width_ratios=[1,5,1,1])
        ax = plt.subplot(gs[0:3, 0:2], projection='skewx')
        skew.draw_title(ax, title)
        ax.grid(True)
        plt.grid(True)

        ax.semilogy(prof.tmpc[~prof.tmpc.mask], prof.pres[~prof.tmpc.mask], 'r', lw=2)
        ax.semilogy(prof.dwpc[~prof.dwpc.mask], prof.pres[~prof.dwpc.mask], 'g', lw=2)
        ax.semilogy(prof.vtmp[~prof.dwpc.mask], prof.pres[~prof.dwpc.mask], 'r--')
        ax.semilogy(prof.wetbulb[~prof.dwpc.mask], prof.pres[~prof.dwpc.mask], 'c-')

        # Plot the parcel trace, but this may fail.  If it does so, inform the user.
        try:
            ax.semilogy(prof.mupcl.ttrace, prof.mupcl.ptrace, 'k--')
        except:
            print("Couldn't plot parcel traces...")

        # Highlight the 0 C and -20 C isotherms.
        ax.axvline(0, color='b', ls='--')
        ax.axvline(-20, color='b', ls='--')

        # Disables the log-formatting that comes with semilogy
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_yticks(np.linspace(100,1000,10))
        ax.set_ylim(1050,100)

        # Plot the hodograph data.
        inset_axes = skew.draw_hodo_inset(ax, prof)
        skew.plotHodo(inset_axes, prof.hght, prof.u, prof.v, color='r')
        #inset_axes.text(srwind[0], srwind[1], 'RM', color='r', fontsize=8)
        #inset_axes.text(srwind[2], srwind[3], 'LM', color='b', fontsize=8)

        # Draw the wind barbs axis and everything that comes with it.
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.set_xlim(-50,50)
        ax2 = plt.subplot(gs[0:3,2])
        ax3 = plt.subplot(gs[3,0:3])
        skew.plot_wind_axes(ax2)
        skew.plot_wind_barbs(ax2, prof.pres, prof.u, prof.v)
        srwind = params.bunkers_storm_motion(prof)
        gs.update(left=0.05, bottom=0.05, top=0.95, right=1, wspace=0.025)

        # Calculate indices to be shown.  More indices can be calculated here using the tutorial and reading the params module.
        p1km = interp.pres(prof, interp.to_msl(prof, 1000.))
        p6km = interp.pres(prof, interp.to_msl(prof, 6000.))
        sfc = prof.pres[prof.sfc]
        sfc_1km_shear = winds.wind_shear(prof, pbot=sfc, ptop=p1km)
        sfc_6km_shear = winds.wind_shear(prof, pbot=sfc, ptop=p6km)
        srh3km = winds.helicity(prof, 0, 3000., stu = srwind[0], stv = srwind[1])
        srh1km = winds.helicity(prof, 0, 1000., stu = srwind[0], stv = srwind[1])
        scp = params.scp(prof.mupcl.bplus, prof.right_esrh[0], prof.ebwspd)
        stp_cin = params.stp_cin(prof.mlpcl.bplus, prof.right_esrh[0], prof.ebwspd, prof.mlpcl.lclhght, prof.mlpcl.bminus)
        stp_fixed = params.stp_fixed(prof.sfcpcl.bplus, prof.sfcpcl.lclhght, srh1km[0], utils.comp2vec(prof.sfc_6km_shear[0], prof.sfc_6km_shear[1])[1])
        ship = params.ship(prof)

        # Setting a dictionary that is a collection of all of the indices we'll be showing on the figure.
        # the dictionary includes the index name, the actual value, and the units.
        indices = {'SBCAPE': [_fmt(prof.sfcpcl.bplus), 'J/kg'],\
                'SBCIN': [_fmt(prof.sfcpcl.bminus), 'J/kg'],\
                'SBLCL': [_fmt(prof.sfcpcl.lclhght), 'm AGL'],\
                'SBLFC': [_fmt(prof.sfcpcl.lfchght), 'm AGL'],\
                'SBEL': [_fmt(prof.sfcpcl.elhght), 'm AGL'],\
                'SBLI': [_fmt(prof.sfcpcl.li5), 'C'],\
                'MLCAPE': [_fmt(prof.mlpcl.bplus), 'J/kg'],\
                'MLCIN': [_fmt(prof.mlpcl.bminus), 'J/kg'],\
                'MLLCL': [_fmt(prof.mlpcl.lclhght), 'm AGL'],\
                'MLLFC': [_fmt(prof.mlpcl.lfchght), 'm AGL'],\
                'MLEL': [_fmt(prof.mlpcl.elhght), 'm AGL'],\
                'MLLI': [_fmt(prof.mlpcl.li5), 'C'],\
                'MUCAPE': [_fmt(prof.mupcl.bplus), 'J/kg'],\
                'MUCIN': [_fmt(prof.mupcl.bminus), 'J/kg'],\
                'MULCL': [_fmt(prof.mupcl.lclhght), 'm AGL'],\
                'MULFC': [_fmt(prof.mupcl.lfchght), 'm AGL'],\
                'MUEL': [_fmt(prof.mupcl.elhght), 'm AGL'],\
                'MULI': [_fmt(prof.mupcl.li5), 'C'],\
                '0-1 km SRH': [_fmt(srh1km[0]), 'm2/s2'],\
                '0-1 km Shear': [_fmt(utils.comp2vec(sfc_1km_shear[0], sfc_1km_shear[1])[1]), 'kts'],\
                '0-3 km SRH': [_fmt(srh3km[0]), 'm2/s2'],\
                '0-6 km Shear': [_fmt(utils.comp2vec(sfc_6km_shear[0], sfc_6km_shear[1])[1]), 'kts'],\
                'Eff. SRH': [_fmt(prof.right_esrh[0]), 'm2/s2'],\
                'EBWD': [_fmt(prof.ebwspd), 'kts'],\
                'PWV': [round(prof.pwat, 2), 'inch'],\
                'K-index': [_fmt(params.k_index(prof)), ''],\
                'STP(fix)': [_fmt(stp_fixed, 'flt'), ''],\
                'SHIP': [_fmt(ship, 'flt'), ''],\
                'SCP': [_fmt(scp, 'flt'), ''],\
                'STP(cin)': [_fmt(stp_cin, 'flt'), '']}

        string = ''
        keys = np.sort(list(indices.keys()))
        x = 0
        counter = 0
        for key in keys:
            string = string + key + ': ' + str(indices[key][0]) + ' ' + indices[key][1] + '\n'
            if counter < 7:
                counter += 1
                continue
            else:
                counter = 0
                ax3.text(x, 1, string, verticalalignment='top', transform=ax3.transAxes, fontsize=11)
                string = ''
                x += 0.3
        ax3.text(x, 1, string, verticalalignment='top', transform=ax3.transAxes, fontsize=11)
        ax3.set_axis_off()

        # Finalize the image formatting and alignments, and save the image to the file.
        if not(filename is None):
            print("SHARPpy Skew-T image output at: ", filename)
            plt.savefig(filename, bbox_inches='tight', dpi=180)
            plt.close()
        else:
            plt.show()
