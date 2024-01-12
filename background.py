import numpy as np
import matplotlib.pyplot as plt 
import astropy.units as u #could be useful
import os 
from scipy.interpolate import interp1d
# import xarray as xr
# import numpy.ma as ma 

class Cmd():
    datafile ={'L':'photometry_brown_dwarf_L.dat', #this one is unused
               'T':'photometry_brown_dwarf_T.dat', #this one is unused
               'MD':'photometry_dupuy_M.dat',
               'LD':'photometry_dupuy_L.dat',
               'TD':'photometry_dupuy_T.dat',
               'MKO_J':'mko_j.txt',
               'MKO_K':'mko_k.txt',
               'MKO_H':'mko_h.txt'
            }
    def __init__(self,datapath='data/') -> None:
        self.datapath = datapath
        self.radius = 1*u.Rjup.to(u.m) #faute de mieux pour l'instant
        self.Jfilter = np.loadtxt(os.path.join(self.datapath,self.datafile['MKO_J']),
                             skiprows=2)
        self.Kfilter = np.loadtxt(os.path.join(self.datapath,self.datafile['MKO_K']),
                             skiprows=2)
        self.Hfilter = np.loadtxt(os.path.join(self.datapath,self.datafile['MKO_H']),
                             skiprows=2)
        self.fluxJVega = None 
        self.fluxHVega = None 
        self.fluxKVega = None
        self._Vega() # Initialise the flux*Vega
        self.magJ_planck = None 
        self.magH_planck = None
        self.magK_planck = None
        self.MD = None 
        self.LD = None 
        self.TD = None
        
    def _Vega(self):
        D =7.76*3.085939e16 #distance vega
        self.fluxJVega = (3.01e-9*4*np.pi*D**2)*self.Jfilter[1:,1]*np.diff(self.Jfilter[:,0])
        self.fluxKVega = (4.00e-10*4*np.pi*D**2)*self.Kfilter[1:,1]*np.diff(self.Kfilter[:,0])
        self.fluxHVega = (1.18e-9*4*np.pi*D**2)*self.Hfilter[1:,1]*np.diff(self.Hfilter[:,0])
    
    def _planck(self):
        """Computes the J, H, K Vega mag for a blackbody
        of temperature between 700 and 2200 K, spaced every 100 K
        """
        nu = np.linspace(1,3,800)*u.um # [1:3] micron
        Teff =  np.array([700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2200])
        
        ## Few useful constants 
        # D=7.76*3.085939e16 #distance vega
        h=6.62607004e-34 #planck
        k=1.380648e-23 #boltxman
        c=299792458 #light speed
        # sigma=5.670373e-8 #stefan-boltzman
        
        ## Loading MKO filters and Vega mag
        Jfilter = self.Jfilter
        Kfilter = self.Kfilter
        Hfilter = self.Hfilter
        
        fluxJVega = self.fluxJVega
        fluxKVega = self.fluxKVega
        fluxHVega = self.fluxHVega
        
        ## Variables initialisation
        planck = np.zeros((Teff.size,nu.size))
        fluxJplanck = np.zeros((Teff.size,Jfilter.shape[0]-1))
        fluxKplanck = np.zeros((Teff.size,Kfilter.shape[0]-1))
        fluxHplanck = np.zeros((Teff.size,Hfilter.shape[0]-1))
        
        ##Actual computation
        for tt in range(Teff.size):
            planck[tt,:] = 2*np.pi*2*h*c**2/(1.e-6*nu.value)**5/(np.exp(h*c/(1.e-6*nu.value)/k/Teff[tt])-1)*1.e-6
            
            #interpolation on each filter
            finterp = interp1d(nu,planck[tt,:])
            fluxJplanck[tt,...] = finterp(Jfilter[1:,0])*Jfilter[1:,1]*4*np.pi*self.radius**2*np.diff(Jfilter[:,0])
            fluxKplanck[tt,...] = finterp(Kfilter[1:,0])*Kfilter[1:,1]*4*np.pi*self.radius**2*np.diff(Kfilter[:,0])
            fluxHplanck[tt,...] = finterp(Hfilter[1:,0])*Hfilter[1:,1]*4*np.pi*self.radius**2*np.diff(Hfilter[:,0])
            
        ## Magnitude computation (with regards to Vega's)
        self.magJ_planck = -2.5*np.log10(np.mean(fluxJplanck,axis=1)/np.mean(fluxJVega))
        self.magK_planck = -2.5*np.log10(np.mean(fluxKplanck,axis=1)/np.mean(fluxKVega))
        self.magH_planck = -2.5*np.log10(np.mean(fluxHplanck,axis=1)/np.mean(fluxHVega))
        
    def _dupuy(self):
        ## Let's deal with the M dwarfs first
        dataMD = np.loadtxt(os.path.join(self.datapath,self.datafile['MD']))
        self.MD = {'magJ':dataMD[:,2]-dataMD[:,0],
                   'magH':dataMD[:,3]-dataMD[:,0],
                   'magK':dataMD[:,4]-dataMD[:,0]
                }
        ## L dwarfs
        dataLD = np.loadtxt(os.path.join(self.datapath,self.datafile['LD']))
        self.LD = {'magJ':dataLD[:,2]-dataLD[:,0],
                   'magH':dataLD[:,3]-dataLD[:,0],
                   'magK':dataLD[:,4]-dataLD[:,0]
                }
        ## T dwarfs
        dataTD = np.loadtxt(os.path.join(self.datapath,self.datafile['TD']))
        self.TD = {'magJ':dataTD[:,2]-dataTD[:,0],
                   'magH':dataTD[:,3]-dataTD[:,0],
                   'magK':dataTD[:,4]-dataTD[:,0]
                }
    
    def _cmd_jk(self):
        if not hasattr(self.magJ_planck,'__len__') or not hasattr(self.MD,'__len__'):
            raise PermissionError("You're calling a private method here.\n You need to call method: CMD('jk')")
        fig,ax = plt.subplots(1)
        ##Dupuy data plot
        ax.plot(self.MD['magJ']-self.MD['magK'],
                self.MD['magJ'],
                'o',
                color='k',
                label = "M dwarfs")
        ax.plot(self.LD['magJ']-self.LD['magK'],
                self.LD['magJ'],
                'o',
                color='red',
                label = "L dwarfs")
        ax.plot(self.TD['magJ']-self.TD['magK'],
                self.TD['magJ'],
                'o',
                color='blue',
                label = "T dwarfs")
        ## black body now 
        ax.plot(self.magJ_planck-self.magK_planck,
                self.magJ_planck,
                '+-',
                color='k',
                label ='Blackbody')
        ## ax params
        ax.invert_yaxis()
        ax.grid()
        ax.legend()
        ax.set_xlabel("J-K")
        ax.set_ylabel("M$_{\\rm J}$")
        ax.set_xlim(-3,5)
        ax.set_ylim(20,9)
        return fig

    def _cmd_jh(self):
        """plots a J-H CMD
        with a BB and data from Dupui 2012
        Raises:
            PermissionError: private method, should not be called from outside

        Returns:
            matplotlib.figure: the CMD plot
        """
        if not hasattr(self.magJ_planck,'__len__') or not hasattr(self.MD,'__len__'):
            raise PermissionError("You're calling a private method here.\n You need to call method: CMD('jk')")
        fig,ax = plt.subplots(1)
        ##Dupuy data plot
        ax.plot(self.MD['magJ']-self.MD['magH'],
                self.MD['magJ'],
                'o',
                color='k',
                label = "M dwarfs")
        ax.plot(self.LD['magJ']-self.LD['magH'],
                self.LD['magJ'],
                'o',
                color='red',
                label = "L dwarfs")
        ax.plot(self.TD['magJ']-self.TD['magH'],
                self.TD['magJ'],
                'o',
                color='blue',
                label = "T dwarfs")
        ## black body now 
        ax.plot(self.magJ_planck-self.magH_planck,
                self.magJ_planck,
                '+-',
                color='k',
                label ='Blackbody')
        ## ax params
        ax.invert_yaxis()
        ax.grid()
        ax.legend()
        ax.set_xlabel("J-H")
        ax.set_ylabel("M$_{\\rm J}$")
        ax.set_xlim(-3,5)
        ax.set_ylim(20,9)
        return fig
    
    def _cmd_hk(self):
        """plots a H-K CMD
        with a BB and data from Dupui 2012
        Raises:
            PermissionError: private method, should not be called from outside

        Returns:
            matplotlib.figure: the CMD plot
        """
        if not hasattr(self.magJ_planck,'__len__') or not hasattr(self.MD,'__len__'):
            raise PermissionError("You're calling a private method here.\n You need to call method: CMD('jk')")
        fig,ax = plt.subplots(1)
        ##Dupuy data plot
        ax.plot(self.MD['magH']-self.MD['magK'],
                self.MD['magH'],
                'o',
                color='k',
                label = "M dwarfs")
        ax.plot(self.LD['magH']-self.LD['magK'],
                self.LD['magH'],
                'o',
                color='red',
                label = "L dwarfs")
        ax.plot(self.TD['magH']-self.TD['magK'],
                self.TD['magH'],
                'o',
                color='blue',
                label = "T dwarfs")
        ## black body now 
        ax.plot(self.magH_planck-self.magK_planck,
                self.magH_planck,
                '+-',
                color='k',
                label ='Blackbody')
        ## ax params
        ax.invert_yaxis()
        ax.grid()
        ax.legend()
        ax.set_xlabel("H-K")
        ax.set_ylabel("M$_{\\rm K}$")
        ax.set_xlim(-3,5)
        ax.set_ylim(20,9)
        return fig
        
    def plot(self,bands='jk'):
        self._planck()
        self._dupuy()
        # print(self.magH_planck)
        if bands.lower()=='jk':
            self._cmd_jk()
        elif bands.lower()=='jh':
            self._cmd_jh()
        elif bands.lower()=='hk':
            self._cmd_hk()
        elif bands.lower()=='all':
            self._cmd_jk()
            self._cmd_jh()
            self._cmd_hk()
        else:
            raise NotImplementedError("Come back soon")
            
        plt.show()
        
if __name__=="__main__":
    # path = '/home/lteinturier/Documents/PhD/BD/CMD/benjamin sutff/brown-dwarf-diagram' 
    truc = Cmd()
    truc.plot('all')