import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d
import xarray as xr
import numpy.ma as ma 
import re 
from matplotlib.lines import Line2D
from background import Cmd

class Gcm_cmd(Cmd):
    colors = {'1500':'k',
              '1400':'lightcoral',
              '1300':'chocolate',
              '1200':'darkorange',
              '1100':'darkolivegreen',
              '1000':'deepskyblue',
              '900':'midnightblue'
            }
    
    def __init__(self, gcm,datapath='data/') -> None:
        super().__init__(datapath)
        if not isinstance(gcm,list):
            self.gcm = [gcm]
        else:
            self.gcm = gcm #list of str or Path
        self.magGCM = {}
        
    def _loadOne(self,simfile):
        """Loads a GCM spectral file 

        Args:
            simfile (str or Path): name or path to the gcm datafile

        Returns:
            (ndarray): wavelength, planetary-integrated flux (in micron, in per micron)
        """
        
        ## Load from the NetCDF file
        ds = xr.open_dataset(simfile,decode_times=False)
        wn = ds.IR_Wavenumber.values ## in cm-1
        width = ds.IR_Bandwidth.values ## in cm-1
        flux = ds.OLR3D.mean("time_counter").values ## W/m2/cm-1
        area = ds.area.values
        ds.close()
        
        ## integration of the flux on the sphere
        flux = np.sum(flux*area,axis = (1,2))/np.sum(area)
                    
        ## Convert to wavelength space, in micron
        wl = 1.e4/wn
        flux *=(1.e-4*wn**2)
        
        return wl,flux
        
        
    def _bandComputation(self,wl,flux,filter='J'):
        """Computes the Vega absolute magnitude 
        for given filter.

        Args:
            wl (ndarray): wavelength array (in micron)
            flux (ndarray): flux array
            dwl (ndaray): width of spectral bands (in micron)
            filter (str, optional): Filter on which the computation is done. Defaults to 'J'.

        Raises:
            NotImplemented: If filter is not J,K or H

        Returns:
            floats: the J,H or K absolute magnitude
        """
        ## Get the right filter and right Vega flux
        if filter.upper()=='J':
            band = self.Jfilter
            vega = self.fluxJVega
        elif filter.upper()=='H':
            band = self.Hfilter
            vega = self.fluxHVega
        elif filter.upper()=='K':
            band = self.Kfilter
            vega = self.fluxKVega
        else:
            raise NotImplemented("Don't know this filter for now")
                
        ## Interpolate the flux onto the filter
        func = interp1d(wl,flux) #interpolator 
        val = func(band[1:,0])*band[1:,1]*4*np.pi*self.radius**2*np.diff(band[:,0])
        return -2.5 *np.log10(np.mean(val)/np.mean(vega))
    
    def _processOne(self,simfile):
        """Loads a gcm simulation and computes
        its J, H, K absolute mag

        Args:
            simfile (str or Path): path or name of the simulation data

        Returns:
            dic (float): J, H and K absolute magnitudes
        """
        wl,flux = self._loadOne(simfile)
        magj =self._bandComputation(wl,flux,'J')
        magh = self._bandComputation(wl,flux,'H')
        magk = self._bandComputation(wl,flux,'K')
        return {'J':magj,'H':magh,'K':magk}
    
    def _processGCM(self):
        """Process all the GCM simulation 
        and computes their J, H, K absolute mag
        """
        markers = {'20':'s','15':'v','10':'D','6':'o','3':'^'}
        nsimu = len(self.gcm) #numbers of simulations given in input
        if nsimu==0:
            raise ValueError("You've given me an empty list of simulations !")
        first = True
        for mysimu in self.gcm: #iteration on all the simulation files
            if first:
                print("INFO: We're time averaging the flux.")
                print("INFO: Check that this is what you want to do")
                first = False
            mags = self._processOne(mysimu)
            ##hypothesis is done here: the pattern "Teq\d+" is in mysimu
            try:
                key = re.search(r"Teq\d+",mysimu).group().replace("Teq","")
            except AttributeError():
                raise SyntaxError("""the pattern 'Teq\d+' is not in the name of the gcm files\n 
                                  Please fix that and relaunch""")
            ##adding a little smthing to get a different marker when different particles size
            try: 
                size = re.search(r"\d+um",mysimu).group().replace("um","")
                mags['symbol'] = markers[size]
            except AttributeError:
                mags['symbol']='+' #default shitty one
            id = key+"_"+size
            self.magGCM[id] = mags
            
        print("INFO: Done processing the GCM simulations")
    
    def _cmd_jk(self):
        """Plots a J-K CMD with a BB, data from Dupuy 2012
        and the GCM simulations

        Raises:
            PermissionError: private method, should not be called from outside.

        Returns:
            matplotlib.figure: the CMD plot
        """
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
        
        ## Plotting the GCM simulations
        for key in self.magGCM.keys():
            col = self.colors[key.split("_")[0]]
            ax.plot(self.magGCM[key]['J']-self.magGCM[key]['K'],
                    self.magGCM[key]['J'],
                    self.magGCM[key]['symbol'],
                    color=col,
                    markersize=12
                )
        ##legend
        markerlegend = [
                    Line2D([0],[0],color='k',markerfacecolor='w',marker='s',linestyle='None',label='20 $\\mathrm{\\mu}$m'),
                    Line2D([0],[0],color='k',markerfacecolor='w',marker='v',linestyle='None',label='15 $\\mathrm{\\mu}$m'),
                    Line2D([0],[0],color='k',markerfacecolor='w',marker='D',linestyle='None',label='10 $\\mathrm{\\mu}$m'),
                    Line2D([0],[0],color='k',markerfacecolor='w',marker='o',linestyle='None',label='6 $\\mathrm{\\mu}$m'),
                    Line2D([0],[0],color='k',markerfacecolor='w',marker='^',linestyle='None',label='3 $\\mathrm{\\mu}$m'),
                    ]
        templegend = [Line2D([0],[0],color=self.colors[key],label=key+' K') for key in self.colors.keys()]
        bigleg = markerlegend+templegend
        h,_ = ax.get_legend_handles_labels()
        ax.legend(handles=bigleg+h,ncol=len(bigleg+h)//3,loc='lower center',bbox_to_anchor=(0.5,0.9))
        
        ## ax params
        ax.invert_yaxis()
        ax.grid()
        ax.set_xlabel("J-K")
        ax.set_ylabel("M$_{\\rm J}$")
        ax.set_xlim(-3,5)
        ax.set_ylim(20,9)
        return fig
    
    def _cmd_jh(self):
        """Plots a J-H CMD with a BB, data from Dupuy 2012
        and the GCM simulations

        Raises:
            PermissionError: private method, should not be called from outside.

        Returns:
            matplotlib.figure: the CMD plot
        """
        if not hasattr(self.magJ_planck,'__len__') or not hasattr(self.MD,'__len__'):
            raise PermissionError("You're calling a private method here.\n You need to call method: CMD('jh')")
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
        
        ## Plotting the GCM simulations
        for key in self.magGCM.keys():
            col = self.colors[key.split("_")[0]]
            ax.plot(self.magGCM[key]['J']-self.magGCM[key]['H'],
                    self.magGCM[key]['J'],
                    self.magGCM[key]['symbol'],
                    color=col,
                    markersize=12,
                )
            
        ##legend
        markerlegend = [Line2D([0],[0],color='k',markerfacecolor='w',marker='s',linestyle='None',label='20 $\\mathrm{\\mu}$m'),
                    Line2D([0],[0],color='k',markerfacecolor='w',marker='v',linestyle='None',label='15 $\\mathrm{\\mu}$m'),
                    Line2D([0],[0],color='k',markerfacecolor='w',marker='D',linestyle='None',label='10 $\\mathrm{\\mu}$m'),
                    ]
        templegend = [Line2D([0],[0],color=self.colors[key],label=key+' K') for key in self.colors.keys()]
        bigleg = markerlegend+templegend
        h,_ = ax.get_legend_handles_labels()
        ax.legend(handles=bigleg+h,ncol=len(bigleg+h)//3,loc='lower center',bbox_to_anchor=(0.5,0.9))
        
        ## ax params
        ax.invert_yaxis()
        ax.grid()
        ax.set_xlabel("J-H")
        ax.set_ylabel("M$_{\\rm J}$")
        ax.set_xlim(-3,5)
        ax.set_ylim(20,9)
        return fig

    def _cmd_hk(self):
        """Plots a H-K CMD with a BB, data from Dupuy 2012
        and the GCM simulations

        Raises:
            PermissionError: private method, should not be called from outside.

        Returns:
            matplotlib.figure: the CMD plot
        """
        if not hasattr(self.magH_planck,'__len__') or not hasattr(self.MD,'__len__'):
            raise PermissionError("You're calling a private method here.\n You need to call method: CMD('jh')")
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
        
        ## Plotting the GCM simulations
        for key in self.magGCM.keys():
            col = self.colors[key.split("_")[0]]
            ax.plot(self.magGCM[key]['H']-self.magGCM[key]['K'],
                    self.magGCM[key]['H'],
                    self.magGCM[key]['symbol'],
                    color=col,
                    markersize=12
                )
            
        ##legend
        markerlegend = [Line2D([0],[0],color='k',markerfacecolor='w',marker='s',linestyle='None',label='20 $\\mathrm{\\mu}$m'),
                    Line2D([0],[0],color='k',markerfacecolor='w',marker='v',linestyle='None',label='15 $\\mathrm{\\mu}$m'),
                    Line2D([0],[0],color='k',markerfacecolor='w',marker='D',linestyle='None',label='10 $\\mathrm{\\mu}$m'),
                    ]
        templegend = [Line2D([0],[0],color=self.colors[key],label=key+' K') for key in self.colors.keys()]
        bigleg = markerlegend+templegend
        h,_ = ax.get_legend_handles_labels()
        ax.legend(handles=bigleg+h,ncol=len(bigleg+h)//3,loc='lower center',bbox_to_anchor=(0.5,0.9))
        
        ## ax params
        ax.invert_yaxis()
        ax.grid()
        ax.set_xlabel("H-K")
        ax.set_ylabel("M$_{\\rm H}$")
        ax.set_xlim(-3,5)
        ax.set_ylim(20,9)
        return fig

    
    def plot(self,bands='jk'):
        self._planck()
        self._dupuy()
        self._processGCM()
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
    import glob
    gcmfile = sorted(glob.glob("../from_mesopsl/solar/Teq*/10um/highres/XspecIR.nc"))+sorted(glob.glob("../from_mesopsl/solar/Teq*/20um/highres/XspecIR.nc"))+sorted(glob.glob("../from_mesopsl/solar/Teq*/6um/highres/XspecIR.nc"))#+sorted(glob.glob("../from_mesopsl/solar/Teq*/15um/highres/XspecIR.nc"))
    gcmfile+=sorted(glob.glob("../from_mesopsl/solar/Teq*/3um/highres/XspecIR.nc"))
    # gcmfile = sorted(glob.glob("../from_mesopsl/solar/Teq*/6um/highres/XspecIR.nc"))+sorted(glob.glob("../from_mesopsl/solar/Teq*/10um/highres/XspecIR.nc"))+sorted(glob.glob("../from_mesopsl/solar/Teq*/20um/highres/XspecIR.nc"))
    obj = Gcm_cmd(gcm=gcmfile)
    obj.plot('jk')
    