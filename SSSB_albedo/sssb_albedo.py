import numpy as np
from numpy import cos, sin, tan, arccos, arcsin, arctan, arctan2, pi, exp, sqrt, cosh, fabs

from math import erf

from numba import jit, njit, prange

from scipy.interpolate import griddata

import matplotlib.pyplot as plt
from matplotlib import colors

import time
from astropy.time import Time
from astropy.coordinates import SkyCoord

import pandas as pd

from tqdm.autonotebook import tqdm

from astroquery.jplhorizons import Horizons


class sssb(object):

    def __init__(self,
                 source,
                 time,
                 frame,
                 pixelsize=1.0,
                 srange=10.0,
                 n_los_steps=400):

        self.source = source
        self.time   = time

        self.frame  = frame
        self.pixelsize = pixelsize
        self.srange = srange
        self.n_los_steps = n_los_steps
        
        self.define_sky()

        self.los = num_los_setup(pixelsize=self.pixelsize,
                                 srange=self.srange,
                                 n_los_steps=self.n_los_steps)
        
        self.image = self.los_source()
        
        
        
    def define_sky(self):

        # definition of image space
        # minmax range (similar to above but now also including s)
        self.phi_min,self.phi_max = -180,180
        self.theta_min,self.theta_max = -90,90
        self.deg2rad = pi/180

        # definition of pixel size and number of bins
        self.n_phi = int((self.phi_max-self.phi_min)/self.pixelsize)
        self.n_theta = int((self.theta_max-self.theta_min)/self.pixelsize)

        self.theta_arrg = np.linspace(self.theta_min,self.theta_max,self.n_theta+1)*self.deg2rad
        self.phi_arrg = np.linspace(self.phi_min,self.phi_max,self.n_phi+1)*self.deg2rad
        self.theta_arr = (self.theta_arrg[1:]+self.theta_arrg[0:-1])/2
        self.phi_arr = (self.phi_arrg[1:]+self.phi_arrg[0:-1])/2

        # define 2D meshgrid for image coordinates
        self.PHI_ARRg, self.THETA_ARRg = np.meshgrid(self.phi_arrg,self.theta_arrg)
        self.PHI_ARR, self.THETA_ARR = np.meshgrid(self.phi_arr,self.theta_arr)

        # jacobian (integral measure on a sphere, exact for this pixel definition, should be 4pi)
        self.dOmega = (np.sin(self.THETA_ARR + self.pixelsize/2) - np.sin(self.THETA_ARR - self.pixelsize/2)) * self.pixelsize

        

    def los_source(self):

        if self.source == 'MBA solid':

            return los_torus(self.PHI_ARR,
                             self.THETA_ARR,
                             1,0,0, # values for MBA: these need to change depending on time and assumptions
                             1,
                             0,0,0,
                             2.8,
                             0.5,
                             frame=self.frame)


        elif self.source == 'MBA smooth':

            #self.los = num_los_setup()

            self.los.los_Gaussian_Torus(1,0,0, # values for MBA: these needs to change with time and other assumptions
                                        2.8,0,1/6)

            return self.los.torus_map


        elif self.source == 'Kuiper Belt solid':

            pass
            #return los_torus(self)


        elif self.source == 'Kuiper Belt smooth':

            pass
            #return los_torus(self)

            
        elif self.source == 'Jovian trojans':

            pass
            #return los_trojans(self)


        else:

            print('Source {0} not yet implemented.'.format(self.source))




    def trafo_image_EC2GAL(self):

        print('Changing frame to: Galactic')
        
        # trafo to gal. coordinates
        lon,lat = trafo_ec2gal(self.PHI_ARR,
                               self.THETA_ARR,
                               deg=False)

        # interpolate ecliptic torus to new irregular grid
        image_gal = griddata((lon.ravel(), lat.ravel()),                      # new coordinates are not regular any more
                             self.image.ravel(),                              # the image stays the same, just gets re-arranged
                             (self.PHI_ARR.ravel(), self.THETA_ARR.ravel()),  # the representation we want to be the same (regular grid)
                             method='nearest')                                # nearest neighbour interpolation avoids 'edge effects'

        image_gal = image_gal.reshape(self.PHI_ARR.shape) # rebuild the image to be a regular 2D array

        self.frame = 'Galactic'
        self.image = image_gal
        

    def trafo_image_GAL2EC(self):

        print('Changing frame to: Ecliptic')
        
        # trafo to gal. coordinates
        lon,lat = trafo_gal2ec(self.PHI_ARR,
                               self.THETA_ARR,
                               deg=False)

        # interpolate ecliptic torus to new irregular grid
        image_ec = griddata((lon.ravel(), lat.ravel()),                      # new coordinates are not regular any more
                            self.image.ravel(),                              # the image stays the same, just gets re-arranged
                            (self.PHI_ARR.ravel(), self.THETA_ARR.ravel()),  # the representation we want to be the same (regular grid)
                            method='nearest')                                # nearest neighbour interpolation avoids 'edge effects'

        image_ec = image_ec.reshape(self.PHI_ARR.shape) # rebuild the image to be a regular 2D array

        self.frame = 'Ecliptic'
        self.image = image_ec

        

            

    def plot_source(self,projection=None):

        plt.subplot(projection=projection)

        if projection == None:
            r2d = 180/pi
        else:
            r2d = 1
        
        if self.frame == 'Galactic':
            xlabel = 'Galactic Longitude [deg]'
            ylabel = 'Galactic Latitude [deg]'
        else:
            xlabel = 'Ecliptic Longitude [deg]'
            ylabel = 'Ecliptic Latitude [deg]'


        plt.pcolormesh(self.PHI_ARRg*r2d,
                       self.THETA_ARRg*r2d,
                       self.image)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
            






class num_los_setup():

    def __init__(self,
                 pixelsize=1.0,
                 srange=10.0,
                 n_los_steps=400):


        # definition of image space
        # minmax range (similar to above but now also including s)
        self.lmin,self.lmax = -180,180
        self.bmin,self.bmax = -90,90
        self.deg2rad = pi/180

        # definition of pixel size and number of bins
        self.pixelsize = pixelsize
        self.n_l = int((self.lmax-self.lmin)/self.pixelsize)
        self.n_b = int((self.bmax-self.bmin)/self.pixelsize)

        # range for line of sight integration: important depending on case (MBA ~ up to 5 AU, Jupiter up to 10 AU etc.)
        self.srange = srange
        # sun position
        self.x0 = 0.
        self.y0 = 0.
        self.z0 = 0.
        self.smin,self.smax = 0,self.srange

        # use resolutin of 0.025 AU
        # driven here empirical
        self.n_los_steps = n_los_steps

        # define lon, lat, and los arrays, with and without boundaries (sky dimensions)
        self.s = np.linspace(self.smin,self.smax,self.n_los_steps)
        self.ds = np.diff(self.s)[0] # los element
        self.bg = np.linspace(self.bmin,self.bmax,self.n_b+1)
        self.lg = np.linspace(self.lmin,self.lmax,self.n_l+1)
        self.b = (self.bg[1:]+self.bg[0:-1])/2
        self.l = (self.lg[1:]+self.lg[0:-1])/2

        # define 2D meshgrid for image coordinates
        self.L_ARRg, self.B_ARRg = np.meshgrid(self.lg,self.bg)
        self.L_ARR, self.B_ARR = np.meshgrid(self.l,self.b)

        # define solid angle for each pixel for normalisations later
        self.domega = (self.pixelsize*self.deg2rad)*(np.sin(np.deg2rad(self.B_ARR+self.pixelsize/2)) - np.sin(np.deg2rad(self.B_ARR-self.pixelsize/2)))

        # 3D grid for los
        self.grid_s, self.grid_b, self.grid_l = np.meshgrid(self.s,self.b,self.l,indexing="ij")


    def los_Gaussian_Torus(self,
                           x0,y0,z0,
                           RT,Rt,sigmaT):

        self.torus_map_slices = cos(self.deg2rad*self.grid_b)*np.vectorize(Gaussian_Torus)(self.grid_s, self.grid_b, self.grid_l,
                                                                                           x0,y0,z0,
                                                                                           RT,Rt,sigmaT)

        self.torus_map = np.sum(self.torus_map_slices*self.ds,axis=0)
        

        

@jit(nopython=True)
def Gaussian_Torus(s,b,l,
                   x0,y0,z0,
                   RT,Rt,sigmaT):

    deg2rad = pi/180.
    
    # coordinates of the observer in AU from the Sun
    #x0 = 0.
    #y0 = -1.
    #z0 = 0.

    # los vector
    x = x0 + s*cos(deg2rad*l)*cos(deg2rad*b)
    y = y0 + s*sin(deg2rad*l)*cos(deg2rad*b)
    z = z0 + s*sin(deg2rad*b)

    # model parameters for MBA torus
    #RT = 2.8
    #Rt = 0.0
    #sigmaT = 1/6

    Rxy = sqrt(x**2 + y**2)

    # complete torus: (here sigma_t = sigma_T)
    val = exp(-1/(2*sigmaT**2)*((Rxy - RT)**2 + (z - Rt)**2))
    
    return val







        
def los_torus(phi,
              theta,
              x0,y0,z0,
              rhoT,
              xT,yT,zT,
              RT,Rt,
              frame='Ecliptic'):

    torus = vlos_torus(phi,theta,# in rad
                       x0,y0,z0, # Earth / Observer
                       rhoT,     # rho
                       xT,yT,zT, # position of torus (not all values valid with definition above)
                       RT,       # large radius
                       Rt)       # small radius

    if frame == 'Galactic':
        
        # trafo to gal. coordinates
        lon,lat = trafo_ec2gal(phi,theta,deg=False)
        
        # interpolate ecliptic torus to new irregular grid
        torus_gal = griddata((lon.ravel(), lat.ravel()),    # new coordinates are not regular any more
                             torus.ravel(),                 # the image stays the same, just gets re-arranged
                             (phi.ravel(), theta.ravel()),  # the representation we want to be the same (regular grid)
                             method='nearest')              # nearest neighbour interpolation avoids 'edge effects'

        torus_gal = torus_gal.reshape(phi.shape) # rebuild the image to be a regular 2D array
        
        return torus_gal
    
    else:
        
        return torus



def los_torus_solution(phi,theta,x0,y0,z0,rhoT,xT,yT,zT,RT,Rt):
    Delta_x = x0-xT
    Delta_y = y0-yT
    Delta_z = z0-zT
    Delta_r = np.sqrt(Delta_x**2 + Delta_y**2 + Delta_z**2)
    p = Delta_x*np.cos(theta)*np.cos(phi) + Delta_y*np.cos(theta)*np.sin(phi) + Delta_z*np.sin(theta)
    q = np.cos(theta)*(Delta_x*np.cos(phi)+Delta_y*np.sin(phi))
    xi = np.sqrt(Delta_r**2 + RT**2 - Rt**2)
    nu = (4*RT**2*Delta_x**2 + 4*RT**2*Delta_y**2)**(0.25)
    A = 1.
    B = 4*p
    C = 4*p**2 + 2*xi**2 - 4*RT**2*np.cos(theta)**2
    D = 4*p*xi**2 - 8*RT**2*q
    E = xi**4 - nu**4
    
    # can't use vectorisation of np.roots() is used
    # are there analytic solutions to arbitrary quartic equations?
    # use np.vectorize() instead later
    x1,x2,x3,x4 = np.roots([A,B,C,D,E])

    delta = np.arctan2(y0,x0)

    # solution only valid if observer sits inside torus but on in tube
    return np.max([(x1-x2).real,(x3-x4).real])

# because the np.roots() function doesnt take 'multiple polynomials' at the same time
# (i.e. it can only handle one coordinate pair phi/theta at once)
# we vectorise the function, so that it can
vlos_torus = np.vectorize(los_torus_solution)






def trafo_ec2gal(phi,theta,deg=False):
    deg2rad = np.pi/180
    alpha = 60.188*deg2rad
    beta = 96.377*deg2rad
    
    lat = arcsin(-sin(alpha)*cos(phi)*cos(theta) + cos(alpha)*sin(theta))
    
    lon = arctan2(cos(alpha)*sin(beta)*cos(phi)*cos(theta) + sin(alpha)*sin(beta)*sin(theta) + cos(beta)*sin(phi)*cos(theta),
                  cos(alpha)*cos(beta)*cos(phi)*cos(theta) + sin(alpha)*cos(beta)*sin(theta) - sin(beta)*sin(phi)*cos(theta))
    
    if deg == True:
        return lon/deg2rad,lat/deg2rad
    else:
        return lon,lat


def trafo_gal2ec(lon,lat,deg=False):
    deg2rad = np.pi/180
    alpha = 60.188*deg2rad
    beta = 96.377*deg2rad
    
    theta = arcsin(sin(alpha)*cos(beta)*cos(lon)*cos(lat) + sin(alpha)*sin(beta)*sin(lon)*cos(lat) + cos(alpha)*sin(lat))
    
    phi = arctan2(-sin(beta)*cos(lon)*cos(lat) + cos(beta)*sin(lon)*cos(lat),
                  cos(alpha)*cos(beta)*cos(lon)*cos(lat) + cos(alpha)*sin(beta)*sin(lon)*cos(lat) - sin(alpha)*sin(lat))
    
    if deg == True:
        return phi/deg2rad,theta/deg2rad
    else:
        return phi,theta
