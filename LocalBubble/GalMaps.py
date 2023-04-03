# Imports:
from astropy.io import fits
import healpy as hp
import numpy as np
from mhealpy import HealpixMap
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
import pandas as pd
from astropy.wcs import WCS
import astropy.wcs.utils as utils
from astropy.coordinates import SkyCoord
import math

class GalMapsHeal:

    def read_healpix_file(self,input_file,verbose=False):
        
        """
        Read GALPROP map given in healpix format.
        
        Input:
        input_file [str]: input GALPROP map in healpix format.
        verbose: If True print more info about data. Default is False.
        Note: Currently only compatible with GALPROP v57. 
        """

        print()
        print("**********************")
        print("GALPROP HEALPIX READER")
        print()
        
        # Read in galprop map:
        input_file = input_file
        hdu = fits.open(input_file)
        header = hdu[1].header

        # Get map info:
        self.ordering = header["ORDERING"]
        self.nside = header["NSIDE"]
        self.NPIX = hp.nside2npix(self.nside) # number of pixels
        self.resolution = hp.nside2resol(self.nside, arcmin=True) / 60.0 # spatial resolution
        self.resolution = format(self.resolution,'.3f')
        
        # Print info:
        print()
        print("Input file: " + str(input_file))
        print("Ordering: " + str(self.ordering))
        print("NSIDE: " + str(self.nside))
        print("Approximate resolution [deg]: " + str(self.resolution))
        print("Number of pixels: " + str(self.NPIX))
        print()

        # Get data:
        # Note: energy and data are stored as numpy records:
        self.energy = hdu[2].data # MeV
        self.energy = self.energy['energy'] 
        self.num_ebins = self.energy.size
        self.data = hdu[1].data
    
        # Print info verbose:
        if verbose == True:
            print()
            print("Data record:")
            print("Size: " + str(self.data.size))
            print("units: ph/cm^2/s/MeV/sr")
            print(self.data)
            print()
            print("Number of Energy bins: " + str(self.num_ebins))
            print("Energy array [MeV]:")
            print(self.energy)
            print()

        return

    def make_healmap(self,energy_bin):

        """
        Define healpix object for given energy bin.
        
        Input:
        energy_bin [int]: energy slice to use for healpix object. 
        """

        # Get data slice for given energy bin:
        this_ebin = "Bin" + str(energy_bin)
        ebin_data = self.data[this_ebin]
        data_indices=np.arange(len(ebin_data))

        # Define Healpix object:
        self.galmap = HealpixMap(ebin_data, nside = self.nside, scheme = self.ordering)
        
        return

    def plot_healmap(self,savefile,plot_kwargs={},fig_kwargs={}):

        """
        Plot healpix map.

        Input:
        savefile [str]: Name of output image file.
        
        Optional:
        plot_kwargs [dict]: pass any kwargs to plt.imshow().
        fig_kwargs [dict]: pass any kwargs to plt.gca().set().
        """
        
        # Make plot:
        plot,ax = self.galmap.plot(cmap="inferno",cbar=True,**plot_kwargs)#,norm=colors.LogNorm(),cbar=True)
        ax.get_figure().set_figwidth(7)
        ax.get_figure().set_figheight(6)
        plot.colorbar.set_label("$\mathrm{ph \ cm^{-2} \ s^{-1} \ sr^{-1}} \ MeV^{-1}$")
        ax.set(**fig_kwargs)
        plt.savefig(savefile,bbox_inches='tight')
        plt.show()
        plt.close()

        return

    def get_disk_region(self,theta,phi,rad):
    
        """
        Get pixels corresonding to disk region.
        
        Inputs:
        theta: colatitude (zenith angle) in degrees, ranging from 0 - 180 degrees.
        phi: longitude (azimuthal angle) in degrees, ranging from 0 - 360 degrees..
        rad: radius of region in degrees.
        """
       
        # Convert degree to radian:
        theta = np.radians(theta)
        phi = np.radians(phi)
        rad = np.radians(rad)
        
        # Get pixels:
        center_pix = self.galmap.ang2pix(theta,phi,lonlat=False)
        center_vec = self.galmap.pix2vec(center_pix)
        pixs = self.galmap.query_disc(center_vec,rad) 

        return pixs

    def get_polygon_region(self,theta,phi):

        """
        Get pixels corresponding to polygon region. 
        Note: The order of the vertices matters --
              the polygon can't be self-intersecting,
              otherwise an "unknown exception" will be thrown.
        
        Inputs:
        theta: list of theta values for vertices, in degrees.
               Note: theta is the colatitude (zenith angle), 
               ranging from 0 - 180 degrees.
        phi:   list of phi values for vertices, in degrees.
               Note: phi is the longitude, ranging from 0 - 360 degrees.
        """
        
        # Convert lists in degrees to radians:
        # Note: healpy requires radians as input. 
        theta = np.array(theta)
        theta = np.radians(theta)
        phi = np.array(phi)
        phi = np.radians(phi) 
       
        # Make vertex array:
        vertices = []
        for i in range(0,len(theta)):
            center_pix = self.galmap.ang2pix(theta[i],phi[i],lonlat=False)
            center_vec = self.galmap.pix2vec(center_pix)
            vertices.append(list(center_vec))
        vertices = np.array(vertices)
        
        print()
        print("vertex array:")
        print("shape: " + str(vertices.shape))
        print(vertices)
        print()
        
        # Get pixels:
        pixs = self.galmap.query_polygon(vertices,inclusive=True)

        return pixs 
    
    def mask_region(self,pixs):
        
        """
        Mask region given by pixs arguement.
        pixs: healpy pixs to be masked.
        """

        self.galmap[pixs] = 0

        return

    def make_spectrum(self,pixs=None):

        """
        Make average spectrum over specified region.
        
        Optional Inputs:
        pixs [array]: Healpix pixels to use. 
            - Default is None, which uses all-sky.
        """
        
        # Make spectrum:
        spectra_list = []
        for E in range(0,len(self.energy)):
            this_bin = "Bin%s" %str(E)
            
            # If averaging over all-sky:
            if pixs is None:
                spectra_list.append(np.mean(self.data[this_bin]))
            
            # If averaging over limited region:
            else:
                spectra_list.append(np.mean(self.data[this_bin][pixs]))
        
        # Scale of energy:
        self.spectra_list = (self.energy**2)*spectra_list
        
        return

    def write_spectrum(self,savefile):

        """
        Write spectrum to file.
        
        Input:
        savefile [str]: Name of output data file (tab delimited).
        """
        
        # Need to reformat energy data to be commpatible with pandas:
        self.energy = np.array(self.energy).astype("float")

        # Write to file:
        d = {"energy[MeV]":self.energy,"flux[MeV/cm^2/s/sr]":self.spectra_list}
        df = pd.DataFrame(data=d)
        df.to_csv(savefile,float_format='%10.5e',index=False,sep="\t",columns=["energy[MeV]", "flux[MeV/cm^2/s/sr]"])

        return

    def plot_spectrum(self,savefile):

        """
        Plot map spectrum.
        
        Input:
        savefile: Name of saved image file.
        """

        # Setup figure:
        fig = plt.figure(figsize=(9,6))
        ax = plt.gca()
            
        # Plot:
        plt.loglog(self.energy,self.spectra_list)
        
        plt.ylim(1e-5,1e-1)
        plt.xlabel("Energy [MeV]", fontsize=14)
        plt.ylabel("$\mathrm{E^2 \ dN/dE \ [\ MeV \ cm^{-2} \ s^{-1} \ sr^{-1}]}$",fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax.tick_params(axis='both',which='major',length=9)
        ax.tick_params(axis='both',which='minor',length=5)
        plt.savefig(savefile,bbox_inches='tight')
        plt.show()
        plt.close()

        return

    def gal2mega(self,file_type,output_file):

        """
        Convert GALPROP map to MEGAlib cosima input.
        
        Inputs:
        file_type: 'fits' for mapcupe fits file or 'heal' for healpix file.
        output_file: Name of output file (do not include .dat). 
        """
        
        # Make energy array:
        energy_list = []
        for each in self.energy[:21]:
            this_energy = float('{:.1f}'.format(each*1000.0)) # convert to keV and format
            energy_list.append(this_energy)
        print() 
        print("energy list [keV]:")
        print(energy_list)
        print()

        # Define phi (PA), theta (TA), and energy (EA) points:
        PA = np.arange(-179.5,181.5,1)
        TA = np.arange(0,180.5,0.5)
        EA = np.array(energy_list)

        # Convert PA to file input:
        PA_line = "PA"
        for i in range(0,len(PA)):
            PA_line += " " + str(PA[i])

        # Convert TA to file input:
        TA_line = "TA"
        for i in range(0,len(TA)):
            TA_line += " " + str(TA[i])

        # Convert EA to file input:
        EA_line = "EA"
        for i in range(0,len(EA)):
            EA_line += " " + str(EA[i])

        # Write file:
        f = open(output_file + ".dat","w")
        f.write("IP LIN\n")
        f.write(PA_line + "\n")
        f.write(TA_line + "\n")
        f.write(EA_line + "\n")

        # Make main:
        for E in range(0,len(energy_list)):
    
            this_E_list = []
            for i in range(0,len(PA)):
                
                if PA[i] > 0:
                    this_l = PA[i]
                if PA[i] < 0:
                    this_l = 360 + PA[i]

                for j in range(0,len(TA)):
        
                    this_b = 90-TA[j]

                    # to get flux from healpix map:
                    if file_type == "heal":
                        
                        theta = np.radians(TA[j])
                        phi = np.radians(this_l)
                        this_ebin = "Bin%s" %str(E)
                        this_pix = self.galmap.ang2pix(theta,phi,lonlat=False)
                        this_flux = self.data[this_ebin][this_pix] / 1000.0 # ph/cm^2/s/keV/sr

                    # to get flux from mapcube:
                    if file_type == "fits":
                        
                        #pixs = self.wcs.all_world2pix(np.array([[this_l,this_b,0]]),0)
                        pixs = self.wcs.all_world2pix(np.array([[this_l,this_b]]),0)
                        this_l_pix = int(math.floor(pixs[0][0]))
                        this_b_pix = int(math.floor(pixs[0][1]))
                        this_flux = self.data[this_b_pix,this_l_pix]# / 1000.0 # ph/cm^2/s/keV/sr

                    # Format:
                    this_flux = float('{:.5e}'.format(this_flux))

                    # Write line:
                    this_line = "AP " + str(i) + " " + str(j) + " " + str(E) + " " + str(this_flux) + "\n"
                    f.write(this_line)
                   
        # Close file:
        f.write("EN")
        f.close()

        return
    
class GalMapsFits(GalMapsHeal):

    def read_fits_file(self, input_file, energy_list):

        """Read GALPROP map given in fits format."""

        print()
        print("**********************")
        print("GALPROP Fits READER")
        print()

        hdu = fits.open(input_file)
        self.data = hdu[0].data
        #self.energy = hdu[1].data
        #self.energy = self.energy['Energy']
        self.energy = [energy_list]
        header = hdu[0].header
        self.wcs = WCS(header)

        return

    def get_fits_region(self,lat,lon,lon2=[]):

        """
        Get pixels for specified spatial region.
        
        Inputs:
        lat [list]: min and max Galactic latitude of region, inclusive. 
        lon [list]: min and max Galactic longitude of region, inclusive.
        
        Optional:
        lon2 [list]: min and max Galactic longtitude of region (inclusive, exclusive). 
            - This arguement compliments 'lon' for regions about the 
            Galactic center, since l ranges from 0 - 360 degrees. 
        """

        # Make ra and dec lists:
        index_array = np.argwhere(self.data[0,:,:] < 1e50) #arbitrary condition to ensure all pixels are extracted
        ralist = []
        declist = []
        for each in index_array:
            ra = each[1]
            dec = each[0]
            ralist.append(ra)
            declist.append(dec)

        # Transfer pixels in wcs to sky coordinates:
        coords = utils.pixel_to_skycoord(ralist,declist,self.wcs)
    
        # Define keep index:
        
        # longitude main:
        i1 = (coords.l.deg>=lon[0]) & (coords.l.deg<=lon[1])
        
        # A second condition is needed for regions centered on the Galactic center, 
        # since l ranges from 0 - 360 degrees.
        if len(lon2) != 0:
            i2 = (coords.l.deg>=lon2[0]) & (coords.l.deg<lon2[1])
            l_cond = i1 | i2
        if len(lon2) == 0:
            l_cond = i1 
        
        # latitude condition:
        b_cond = (coords.b.deg>=lat[0]) & (coords.b.deg<=lat[1])
        
        # define keep index:
        keep_index = b_cond & l_cond

        # Convert sky coordinates in wcs back to pixels:
        pixs_SR =  utils.skycoord_to_pixel(coords[keep_index],self.wcs,mode="wcs")

        # Need to convert float to int
        # Note: need to round fist, since int rounds towards zero
        SR0 = np.round(pixs_SR[0]).astype(int)
        SR1 = np.round(pixs_SR[1]).astype(int)
     
        return [SR0,SR1]
 
    def make_spectrum(self,pixs=None):

        """
        Make spectrum by averaging over specified region.
        
        Optional:
        pixs [list]: pixels of region to be used for calculation.
            - Should be a list containing SRO and SR1, 
              returned from self.gets_fits_region.
        """

        spectra_list = []
        for E in range(0,len(self.energy)):
            
            # Get average over all-sky:
            if pixs is None:
                spectra_list.append(np.mean(self.data[E,:,:]))

            # Get average over specified region:
            else:
                spectra_list.append(np.mean(self.data[E,pixs[1],pixs[0]]))
        
        spectra_list = np.array(spectra_list)
        self.spectra_list = (self.energy**2)*spectra_list
       
        return

class Utils(GalMapsFits):

    def sum_spectra(self,savefile,input_files):

        """
        Sum multiple spectra. Added files must have the same binning.
        
        inputs:
        savefile [str]: name of saved .dat file.
        input_files [list]: list of spectra to add, given as .dat files.
        """

        # Add spectra from list:
        for i in range(len(input_files)):
            df = pd.read_csv(input_files[i], delim_whitespace=True)
            this_energy = df["energy[MeV]"]
            this_spectra = df["flux[MeV/cm^2/s/sr]"]
            if i == 0:
                summed_spectra = this_spectra
            if i > 0:
                summed_spectra += this_spectra

        # Write to file:
            d = {"energy[MeV]":this_energy,"flux[MeV/cm^2/s/sr]":summed_spectra}
            df = pd.DataFrame(data=d)
            df.to_csv(savefile,float_format='%10.5e',index=False,sep="\t",columns=["energy[MeV]", "flux[MeV/cm^2/s/sr]"])
    
        return

    def plot_mult_spectra(self,savefile,input_files,labels,fig_kwargs={},\
            plot_kwargs={},show_plot=True):

        """
        Plot multiple map spectra.
        
        Inputs:
        savefile [str]: Name of saved image file.
        input_files [list]: List of input file(s) to plot.
        labels [list]: List of legend labels corresponding to input files.
            -Must have a label for each input file.
        
        Optional:
        fig_kwargs [dict]: pass any kwargs to plt.gca().set()
        plot_kwargs [dict]: pass any kwargs to plt.plot().
            - Each key must define a list corresponding to input_files. 
        show_plot: set to False to not show plot. 
        """

        # Setup figure:
        fig = plt.figure(figsize=(9,6))
        ax = plt.gca()
             
        # Check for proper data type:
        if isinstance(input_files,list) == True:
            pass
        else:
            raise TypeError("'input_files' must be a list.")
        
        # Check labels:
        if len(labels) != len(input_files):
            raise ValueError("Must have a label for each input file.")    

        # Plot spectra:
        for i in range(len(input_files)):

            # Check for plot kwargs:
            this_kwargs = {}
            for key in plot_kwargs:
                this_kwargs[key] = plot_kwargs[key][i]
                    
            df = pd.read_csv(input_files[i], delim_whitespace=True)
            this_energy = df["energy[MeV]"]
            this_spectra = df["flux[MeV/cm^2/s/sr]"]
            plt.loglog(this_energy,this_spectra,label=labels[i],**this_kwargs)

        plt.ylim(1e-5,1e-1)
        plt.xlabel("Energy [MeV]", fontsize=14)
        plt.ylabel("$\mathrm{E^2 \ dN/dE \ [\ MeV \ cm^{-2} \ s^{-1} \ sr^{-1}]}$",fontsize=14)
        plt.legend(frameon=True,ncol=1,loc=2,handlelength=2,prop={'size': 9.5})
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax.tick_params(axis='both',which='major',length=9)
        ax.tick_params(axis='both',which='minor',length=5)
        ax.set(**fig_kwargs)
        #plt.grid(ls=":",color="grey",alpha=0.4)
        plt.savefig(savefile,bbox_inches='tight')
        if show_plot == True:
            plt.show()
            plt.close()

        return
