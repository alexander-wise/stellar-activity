#normalize spectra

import numpy as np
from numpy import *
import os
import sys
from scipy.stats.stats import linregress
import scipy.interpolate as interpolate
from astropy.io import fits
from datetime import datetime
from jdcal import gcal2jd
from tqdm import tqdm

#dataset="HARPS-N_solar"
#dataset="AlphaCenB"

mode = 'parallel'
#mode = 'home'

#read in the input parameters
job_type = sys.argv[1] if mode=='parallel' else 'norm' #spec:normalization or ordr:stacking for parallel jobs
proc_local = int(sys.argv[2]) if mode=='parallel' else 0 #local process index
nprocsmax=72 if job_type=='ordr' else 1 if job_type=='norm' else 800

datasetCode = sys.argv[3] if mode=='parallel' else 'ACB'

if datasetCode=="ACB":
	dataset = "AlphaCenB"

if datasetCode=="HNS":
	dataset = "HARPS-N_solar"

#datadir = "/lustre/work/phys/aww/spectra/AlphaCen/"  if mode=='parallel' else "/Volumes/My_Passport/HARPS/AlphaCen/"
#datadir = "/lustre/work/phys/aww/spectra/HARPS/"  if mode=='parallel' else "/Volumes/My_Passport/HARPS/"
if dataset=="HARPS-N_solar":
	datadir = "/gpfs/group/ebf11/default/"
	targets = ["HARPS-N_solar"]

if dataset=="AlphaCenB":
	datadir = "/gpfs/group/ebf11/default/afw5465/AlphaCen/"
	targets = ["B"]


#datadir = "/Users/aww/Desktop/"
#targets = ['tau_ceti_spectra', 'fiber_calibration_spectra']


print 'nprocsmax=' + str(nprocsmax)

print 'Beginning run number ' + str(proc_local)

def xn(x0,p0):
	return poly1d(p0)(x0)


def getWaves(folder, wfile=''):
	waves = array(fits.open(folders[folder]+'waves/'+wfile)[0].data,dtype=np.float64)
	norder, nx = waves.shape
	new_waves = zeros(waves.shape)
	for i in arange(norder):
		waves_fit = polyfit(arange(nx),waves[i],2)
		new_waves[i] = polyval(waves_fit,arange(nx))
	return new_waves


#normalize the spectra and make pictures or normalized data files
def normalizeSpectra(folder):
	#os.system("rm " + folders[folder] + ".DS_Store")
	filelist = array(os.listdir(folders[folder]))
	fitslist = filelist[where(array([('20' in filelist[i][:2]) for i in range(len(filelist))]))[0]]
	fitslist = sort(fitslist)
	#fitslist = delete(fitslist, badspec[folder]) if pics0files1 else fitslist
	ns = len(fitslist)
	if dataset=="HARPS-N_solar":
		norder = fits.open(folders[folder]+fitslist[0]+'/s2d.fits')['SCIDATA'].data.shape[0]
		nx = fits.open(folders[folder]+fitslist[0]+'/s2d.fits')['SCIDATA'].data.shape[-1]
		JDs = genfromtxt(folders[folder] + 'allspNLs.txt')[:,0]
		useSpec = arange(0,len(JDs))
	if dataset=="AlphaCenB":
		JDs = genfromtxt(folders[folder] + 'allspBRVs.txt')[:,0]
		badspec = load(folders[folder]+"badspec.npy")
		goodspec = ones(len(JDs),dtype=bool)
		goodspec[badspec]=0
		JDis = load(folders[folder]+"JDis.npy")
		useSpec = where(goodspec)[0][JDis]
		JDs = JDs[useSpec]
		fitslist = fitslist[useSpec]
		ns = len(fitslist)
		norder = fits.open(folders[folder]+fitslist[0]+'/e2ds.fits')[0].data.shape[0]
		nx = fits.open(folders[folder]+fitslist[0]+'/e2ds.fits')[0].data.shape[-1]
		BERVs = genfromtxt(folders[folder] + 'allspBRVs.txt')[useSpec,1]*1000.0
		BORVs = genfromtxt(folders[folder] + 'allspBRVs.txt')[useSpec,2]*1000.0
		binaryRV = load(folders[folder]+"binaryRV.npy")[useSpec]
		useSpec = arange(0,len(JDs))

	waves_solar = load(folders[folder]+"waves_solar.npy")
	#ws0 = fits.open(folders[folder]+fitslist[-1]+'/s2d.fits')['WAVEDATA_AIR_BARY'].data
	nbox = 100
	nperbox=nx/nbox+1
	p = 2
	tolerance=zeros(norder) + 0.005 #use these values for initializing tolerance determination
	if dataset=="HARPS-N_solar":
		tolerance[:14] += array([ 0.035,  0.0,  0.075,  0.035,  0.015,  0.0,  0.005,  0.0,
        0.0,  0.005,  0.0,  0.0,  0.0,  0.005])
	if dataset=="AlphaCenB":
		tolerance[:28] += array([ 0.035,  0.155,  0.315,  0.315,  0.315,  0.035,  0.075,  0.315,
        0.315,  0.035,  0.155,  0.015,  0.035,  0.005,  0.035,  0.035,
        0.035,  0.015,  0.005,  0.   ,  0.015,  0.   ,  0.035,  0.015,
        0.   ,  0.   ,  0.   ,  0.005])
	maskmax = nbox - 8
	CRmax = 2
	#single process initialization
	processlengths = zeros(nprocsmax, dtype=int) + ns / nprocsmax
	ng_rem = ns - ns / nprocsmax * nprocsmax
	processlengths[:ng_rem] += 1
	proc_i = zeros(nprocsmax, dtype=int)
	for j in range(nprocsmax):
		proc_i[j] = sum(processlengths[:j])
	#loop for this process
	for ii in tqdm(range(proc_i[proc_local],proc_i[proc_local] + processlengths[proc_local])):
		print 'Beginning normalization for spectrum number ' + str(ii)
		fitFailed = zeros(norder)
		ps = zeros((norder,p+1))
		xs = zeros((norder,nbox))
		ys = zeros((norder,nbox))
		if dataset=="HARPS-N_solar":
			fits0 = fits.open(folders[folder]+fitslist[ii]+'/s2d.fits')
			ws = array(fits0['WAVEDATA_AIR_BARY'].data, dtype=float64)
			spectra0 = array(fits0['SCIDATA'].data, dtype=float64) # / blaze #spectra to be normalized
			spectra1 = array(fits0['SCIDATA'].data, dtype=float64) #spectra saved for later RV shifting - unnormalized spectra
			spectra1[where(spectra1<0)]=0.0
			spectraErr = array(fits0['ERRDATA'].data, dtype=float64)
		if dataset=="AlphaCenB":
			fits0 = fits.open(folders[folder]+fitslist[ii]+'/e2ds.fits')[0]
			wfile = fits0.header['HIERARCH ESO DRS CAL TH FILE'][:30] + 'wave_A.fits'
			ws = getWaves(folder,wfile)
			bfile = fits0.header['HIERARCH ESO DRS BLAZE FILE'] #[:30] + 'blaze_A.fits'
			blaze = array(fits.open(folders[folder]+'blazes/'+bfile)[0].data,dtype=np.float64)
			spectra0 = array(fits0.data,dtype=np.float64) / blaze #spectra to be normalized
			fits0 = fits.open(folders[folder]+fitslist[ii]+'/e2ds.fits')[0] #reset this because otherwise modifying spectra1 will modify spectra0
			spectra1 = array(fits0.data,dtype=np.float64) #spectra saved for later RV shifting - unnormalized spectra
			spectra1[where(spectra1<0)]=0.0
		for j in range(norder):
			if len(where(isnan(spectra0[j]))[0]) > 0.5:
				fitFailed[j] = 1.0
			else:
				if dataset=="HARPS-N_solar":
					spectraWeights = spectraErr[j]
				if dataset=="AlphaCenB":
					spectraWeights = sqrt(spectra1[j]) / sum(sqrt(spectra1[j])) #sqrt because numpy.polyfit says weights should be 1/sigma
				lineFitWeights = zeros(nbox)
				for k in range(nbox):
					l = k*nperbox + argmax(spectra0[j][k*nperbox:(k+1)*nperbox])
					xs[j,k] = ws[j][l]
					ys[j,k] = spectra0[j][l]
					lineFitWeights[k] = spectraWeights[l]
				mask = []
				if ((folder,ii,j) in maskrs): #use custom defined mask ranges
					mask = mask + list(where((xs[j] > maskrs[(folder,ii,j)][0]) & (xs[j] < maskrs[(folder,ii,j)][1]))[0])
				x1=delete(xs[j], mask)
				y1=delete(ys[j], mask)
				w1=delete(lineFitWeights, mask)
				ps[j] = polyfit(x1,y1,p,w=w1)
				ssxm, ssxym, ssyxm, ssym =  cov(y1,xn(x1,ps[j]),aweights=w1).flat
				r0 = ssxym / sqrt(ssxm*ssym) #weighted linear regression r-value
				#r0 = linregress(y1,xn(x1,ps[j]))[2] #unweighted linear regression r-value
				CRs = 0
				L=0
				for k in range(nbox/2):
					if k in mask:
						L+=1
				R=0
				for k in range(nbox/2,nbox):
					if k in mask:
						R+=1
				#find the mask ranges automatically
				foundK = True
				while ((abs(r0-1)>tolerance[j]) and (len(mask)<maskmax) and foundK):
					r1 = zeros(nbox) - 10.0
					p1 = zeros((nbox,p+1))
					foundK = False
					for k in range(nbox):
						if (k not in mask):
							if ((ys[j,k] < xn(xs[j,k],ps[j])) or (CRs < CRmax)):
								if ((k in range(nbox/2)) and (L < maskmax/2)) or ((k in range(nbox/2,nbox)) and (R < maskmax/2)):
									foundK = True
									x1=delete(xs[j], mask + [k])
									y1=delete(ys[j], mask + [k])
									w1=delete(lineFitWeights, mask + [k])
									p1[k] = polyfit(x1,y1,p,w=w1)
									ssxm, ssxym, ssyxm, ssym =  cov(y1,xn(x1,p1[k]),aweights=w1).flat
									r1[k] = ssxym / sqrt(ssxm*ssym) #weighted linear regression r-value
									#r1[k] = linregress(y1,xn(x1,p1[k]))[2] #unweighted linear regression r-value
					if foundK:
						l = argmin(abs(r1-1))
						if ys[j,l] > xn(xs[j,l],ps[j]):
							CRs+=1
						mask = mask + [l]
						ps[j] = p1[l]
						r0 = r1[l]
					L=0
					for k in range(nbox/2):
						if k in mask:
							L+=1
					R=0
					for k in range(nbox/2,nbox):
						if k in mask:
							R+=1
				if len(mask) > 0:
					maskis[(folder,ii,j)] = mask
				#repeat the fitting process with the following statement for 1 spectrum until tolerance no longer changes, then use those order-by-order tolerances for all spectra
				#if len(mask) > nbox/1.25:
				#	print("increasing tolerance...")
				#	tolerance[j] *= 2.0
		
		#normalize the spectra where the fit succeeded (all fit function values are > 0)
		for j in range(norder):
			if len(where(xn(ws[j],ps[j])<=0)[0]) == 0:
				spectra0[j] = spectra0[j]/xn(ws[j],ps[j])
			else:
				fitFailed[j]=1.0
		
		#save the normalized spectra to files
		#########################################
		# This may use too much file storage space for large data sets - it could be reduced to only output files containing fit parameters, and the interpolating / RV shifting could be done later
		# To maximize computational speed, the current implementation assumes unlimited file storage space
		#########################################
		spectra2 = zeros(spectra0.shape)
		spectra3 = zeros(spectra0.shape)
		spectra4 = zeros(spectra0.shape)
		spectra5 = zeros(spectra0.shape)
		spectra6 = zeros(spectra0.shape)
		spectra7 = zeros(spectra0.shape)
		sinc_interp = BandLimitedInterpolator()
		order_centers = (ws[:,0]+ws[:,-1]) / 2.0
		solar_order_centers = (waves_solar[:,0]+waves_solar[:,-1]) / 2.0
		if dataset=="HARPS-N_solar":
			RV_offset = 0.0
		if dataset=="AlphaCenB":
			RV_offset = BERVs[ii]-binaryRV[ii]-median(BORVs-binaryRV)
		RVs_2 = zeros(len(JDs)) + RV_offset
		RVs_3 = 0.1*sin((JDs-2455000)*2.*pi/250.+1.7) + RV_offset #10 cm/s amplitude, 250-day period, 1.7 raidian phase shift
		RVs_4 = 0.2*sin((JDs-2455000)*2.*pi/250.+1.7) + RV_offset #20 cm/s amplitude, 250-day period, 1.7 raidian phase shift
		RVs_5 = 0.4*sin((JDs-2455000)*2.*pi/250.+1.7) + RV_offset #40 cm/s amplitude, 250-day period, 1.7 raidian phase shift
		RVs_6 = 0.8*sin((JDs-2455000)*2.*pi/250.+1.7) + RV_offset #80 cm/s amplitude, 250-day period, 1.7 raidian phase shift
		RVs_7 = 0.3*sin((JDs-2455000)*2.*pi/37.+1.7) + 0.6*sin((JDs-2455000)*2.*pi/133.+2.7) + 0.5*sin((JDs-2455000)*2.*pi/365.+3.7) + 0.8*sin((JDs-2455000)*2.*pi/250.+4.7) + RV_offset
		for j in range(norder):
			#spectra2[j] = interp(ws0[j],ws[j],spectra0[j])
			center_diff = amin(abs(order_centers[j]-solar_order_centers))
			k = argmin(abs(order_centers[j]-solar_order_centers))
			if center_diff<10.0:
				spectra2[j] = sinc_interp.interpolate(waves_solar[k],ws[j]*(1.0+RVs_2[ii]/2.99792458e8),spectra0[j])
				spectra3[j] = sinc_interp.interpolate(waves_solar[k],ws[j]*(1.0+RVs_3[ii]/2.99792458e8),spectra0[j])
				spectra4[j] = sinc_interp.interpolate(waves_solar[k],ws[j]*(1.0+RVs_4[ii]/2.99792458e8),spectra0[j])
				spectra5[j] = sinc_interp.interpolate(waves_solar[k],ws[j]*(1.0+RVs_5[ii]/2.99792458e8),spectra0[j])
				spectra6[j] = sinc_interp.interpolate(waves_solar[k],ws[j]*(1.0+RVs_6[ii]/2.99792458e8),spectra0[j])
				spectra7[j] = sinc_interp.interpolate(waves_solar[k],ws[j]*(1.0+RVs_7[ii]/2.99792458e8),spectra0[j])
		ws_1D, sp2_1D = ordersTo1D(waves_solar,spectra2)
		ws_1D, sp3_1D = ordersTo1D(waves_solar,spectra3)
		ws_1D, sp4_1D = ordersTo1D(waves_solar,spectra4)
		ws_1D, sp5_1D = ordersTo1D(waves_solar,spectra5)
		ws_1D, sp6_1D = ordersTo1D(waves_solar,spectra6)
		ws_1D, sp7_1D = ordersTo1D(waves_solar,spectra7)

		save(folders[folder]+fitslist[ii]+'/normRVInterp.npy', spectra2)
		save(folders[folder]+fitslist[ii]+'/wave.npy', ws)
		save(folders[folder]+fitslist[ii]+'/norm.npy', spectra0)
		if sum(fitFailed) > 0.5:
			savetxt(folders[folder]+fitslist[ii]+'/failedFits.txt', where(fitFailed>0.5)[0], fmt='%i')
		
		if ii in useSpec:
			iii = argmin(abs(useSpec-ii))
			save(folders[folder]+"stellar_only/norm"+str(iii)+".npy", array(sp2_1D, dtype=float32))
			save(folders[folder]+"planet10cm/norm"+str(iii)+".npy", array(sp3_1D, dtype=float32))
			save(folders[folder]+"planet20cm/norm"+str(iii)+".npy", array(sp4_1D, dtype=float32))
			save(folders[folder]+"planet40cm/norm"+str(iii)+".npy", array(sp5_1D, dtype=float32))
			save(folders[folder]+"planet80cm/norm"+str(iii)+".npy", array(sp6_1D, dtype=float32))
			save(folders[folder]+"planet_multi/norm"+str(iii)+".npy", array(sp7_1D, dtype=float32))
		
		if ii==0:
			save(folders[folder]+'stellar_only/interp_wavelengths.npy', ws_1D)
			save(folders[folder]+'planet10cm/interp_wavelengths.npy', ws_1D)
			save(folders[folder]+'planet20cm/interp_wavelengths.npy', ws_1D)
			save(folders[folder]+'planet40cm/interp_wavelengths.npy', ws_1D)
			save(folders[folder]+'planet80cm/interp_wavelengths.npy', ws_1D)
			save(folders[folder]+'planet_multi/interp_wavelengths.npy', ws_1D)
			
			save(folders[folder]+'stellar_only/RVs.npy', RVs_2[useSpec])
			save(folders[folder]+'planet10cm/RVs.npy', RVs_3[useSpec])
			save(folders[folder]+'planet20cm/RVs.npy', RVs_4[useSpec])
			save(folders[folder]+'planet40cm/RVs.npy', RVs_5[useSpec])
			save(folders[folder]+'planet80cm/RVs.npy', RVs_6[useSpec])
			save(folders[folder]+'planet_multi/RVs.npy', RVs_7[useSpec])
			
			save(folders[folder]+'stellar_only/JDs.npy', JDs[useSpec])
			save(folders[folder]+'planet10cm/JDs.npy', JDs[useSpec])
			save(folders[folder]+'planet20cm/JDs.npy', JDs[useSpec])
			save(folders[folder]+'planet40cm/JDs.npy', JDs[useSpec])
			save(folders[folder]+'planet80cm/JDs.npy', JDs[useSpec])
			save(folders[folder]+'planet_multi/JDs.npy', JDs[useSpec])




#function by Joe Ninan:
def remove_nans(Y,X=None,method='drop'):
    """ Returns a clean Y data after removing nans in it.
    If X is provided, the corresponding values form X are also matched and (Y,X) is returned.
    Input:
         Y: numpy array
         X: (optional) 1d numpy array of same size as Y
         method: drop: drops the nan values and return a shorter array
                 any scipy.interpolate.interp1d kind keywords: interpolates the nan values using interp1d 
    Returns:
         if X is provided:  (Y,X,NanMask)
         else: (Y,NanMask)
    """
    NanMask = np.isnan(Y)
    if method == 'drop':
        returnY = Y[~NanMask]
        if X is not None:
            returnX = X[~NanMask]
    else: # Do interp1d interpolation
        if X is not None:
            returnX = X
        else:
            returnX = np.arange(len(Y))
        returnY = interpolate.interp1d(returnX[~NanMask],Y[~NanMask],kind=method,fill_value='extrapolate')(returnX)

    if X is not None:
        return returnY,returnX,NanMask
    else:
        return returnY,NanMask

#function by Joe Ninan:
class BandLimitedInterpolator(object):
    """ Interpolator for doing Band-limited interpolation using windowed Sinc function """
    def __init__(self,filter_size = 23, kaiserB=13):
        """ 
        Input:
             filter_size : total number of pixels in the interpolation window 
                           (keep it odd number), default =23
             kaiserB     : beta value for determiniong the width of Kaiser window function
        """
        self.filter_size = filter_size
        self.kaiserB = kaiserB
        self.Filter = self.create_filter_curve(no_of_points = self.filter_size*21)
        self.pixarray = np.arange(-int(self.filter_size/2), int(self.filter_size/2)+1,dtype=np.int)

    def create_filter_curve(self,no_of_points=None):
        """ Returns a cubit interpolator for windowed sinc Filter curve.
        no_of_points: number of intepolation points to use in cubic inteprolator"""
        if no_of_points is None:
            no_of_points = self.filter_size*21
        x = np.linspace(-int(self.filter_size/2), int(self.filter_size/2), no_of_points)
        Sinc = np.sinc(x)
        Window = np.kaiser(len(x),self.kaiserB)
        FilterResponse = Window*Sinc
        # append 0 to both ends far at the next node for preventing cubic spline 
        # from extrapolating spurious values
        return interpolate.CubicSpline( np.concatenate(([x[0]-1],x,[x[-1]+1])), 
                                   np.concatenate(([0],FilterResponse,[0])))

    def interpolate(self,newX,oldX,oldY,PeriodicBoundary=False):
        """ Inteprolates oldY values at oldX coordinates to the newX coordinates.
        Periodic boundary conditions set to True can create worse instbailities at edge..
        oldX and oldY should be larger than filter window size self.filter_size"""
        # First clean and remove any nans in the data
        oldY, oldX, NanMask = remove_nans(oldY,X=oldX,method='linear')
        if np.sum(NanMask) > 0:
            logging.warning('Interpolated {0} NaNs'.format(np.sum(NanMask)))
        oXsize = len(oldX)
        # First generate a 2D array of difference in pixel values
        OldXminusNewX = np.array(oldX)[:,np.newaxis] - np.array(newX)
        # Find the minimum position to find nearest pixel for each each newX
        minargs = np.argmin(np.abs(OldXminusNewX), axis=0)
        # Pickout the those minumum values from 2D array
        minvalues = OldXminusNewX[minargs, range(OldXminusNewX.shape[1])]
        sign = minvalues < 0  # True means new X is infront of nearest old X
        # coordinate of the next adjacent bracketing point
        Nminargs = minargs +sign -~sign
        Nminargs = Nminargs % oXsize  # Periodic boundary
        # In terms of pixel coordinates the shift values will be
        shiftvalues = minvalues/np.abs(oldX[minargs]-oldX[Nminargs])
        # Coordinates to calculate the Filter values
        FilterCoords = shiftvalues[:,np.newaxis] + self.pixarray
        FilterValues = self.Filter(FilterCoords)
        # Coordinates to pick the values to be multiplied with Filter and summed
        OldYCoords = minargs[:,np.newaxis] + self.pixarray
        if PeriodicBoundary:
            OldYCoords = OldYCoords % oXsize  # Periodic boundary
        else:   # Extrapolate the last value till end..
            OldYCoords[OldYCoords >= oXsize] = oXsize-1
            OldYCoords[OldYCoords < 0] = 0

        OldYSlices = oldY[OldYCoords] # old flux values to be multipled with filter values
        return np.sum(OldYSlices*FilterValues,axis=1)






#collapse order-by-order spectra onto 1-D, cutting off overlapping order edges at the central wavelength of the overlapping region
def ordersTo1D(ws0, sp0):
	norder, nx = ws0.shape
	ws = zeros(norder*nx)
	sp = zeros(norder*nx)
	ws[:nx] = ws0[0]
	sp[:nx] = sp0[0]
	for i in range(norder-1):
		if ws0[i,-1] >  ws0[i+1,0]:
			midw = (ws0[i,-1] + ws0[i+1,0]) / 2.0
			midwi = where(ws > midw)[0][0]
			midwi2 = where(ws0[i+1] > midw)[0][0]
			ws[midwi:midwi+nx-midwi2] = ws0[i+1][midwi2:]
			sp[midwi:midwi+nx-midwi2] = sp0[i+1][midwi2:]
		elif ws0[i,-1] ==  ws0[i+1,0]:
			midwi = argmax(ws)
			ws[midwi:midwi+nx] = ws0[i+1]
			sp[midwi:midwi+nx] = sp0[i+1]
		else:
			midwi = argmax(ws)+1
			ws[midwi:midwi+nx] = ws0[i+1]
			sp[midwi:midwi+nx] = sp0[i+1]
	ws = ws[where(ws != 0)]
	sp = sp[where(ws != 0)]
	return ws, sp







"""
#turn the plot data files generated by the code above and combine_order_fits.py into plots
imdir = "/Volumes/My_Passport/HARPS/AlphaCen/B/fitted_orders/n"
outdir = "/Users/aww/Desktop/fitted_orders/n"
n = [0,1]
o = range(72)
s = {}
s[0] = range(104)
s[1] = range(9695)
ms = zeros(9695)

matplotlib.interactive(False)
for i in [1]:
	for j in [67]:
		S = load(imdir+str(i)+'o'+str(j)+'S.npy') #.reshape(len(s[i]),4096,2)
#		M = load(imdir+str(i)+'o'+str(j)+'M.npy') #.reshape(len(s[i]),100,2)
		F = load(imdir+str(i)+'o'+str(j)+'F.npy') #.reshape(len(s[i]),100,2)
		for k in s[i]:
#		for k in [6512]:
#			clf()
#			plot(S[k][:,0],S[k][:,1],'g-')
#			plot(M[k][:,0][M[k][:,0] != 0],M[k][:,1][M[k][:,0] != 0],'k.')
#			plot(F[k][:,0],F[k][:,1],'k-')
#			savefig(outdir+str(i)+'o'+str(j)+'s'+str(k)+'.png')
			ms[k] = (F[k][-1,1] - F[k][0,1]) / (F[k][-1,0]-F[k][0,0]) / median(S[k][:,1])
matplotlib.interactive(True)

#percentile(ms,10)
"""

"""
#redo the doppler shifts for normalized spectra, using norm.npy files to create normRV.npy files
#This function uses old file names from a previous version, it has not been updated so it probably won't work
def dopplerShiftSpectra(folder):
	os.system("rm " + folders[folder] + ".DS_Store")
	filelist = array(os.listdir(folders[folder]))
	fitslist = filelist[where(array([('20' in filelist[i][:2]) for i in range(len(filelist))]))[0]]
	fitslist = sort(fitslist)
#	fitslist = delete(fitslist, badspec[folder]) if pics0files1 else fitslist
	ns = len(fitslist)
	norder = fits.open(folders[folder]+fitslist[0]+'/e2ds.fits')[0].data.shape[0]
	nx = fits.open(folders[folder]+fitslist[0]+'/e2ds.fits')[0].data.shape[-1]
	#get observation days
	JDs = genfromtxt(folders[folder] + 'allspBRVs.txt')[:,0]
	#get radial velocities
	BERVs = genfromtxt(folders[folder] + 'allspBRVs.txt')[:,1]
	BORVs = genfromtxt(folders[folder] + 'allspBRVs.txt')[:,2]
	#use the wavelength scale of the last spectrum as my RV-interpolation domain
	wfile0 = fits.open(folders[folder]+fitslist[-1]+'/e2ds.fits')[0].header['HIERARCH ESO DRS CAL TH FILE'][:30] + 'wave_A.fits'
	ws0 = fits.open(folders[folder]+'waves/'+wfile0)[0].data
	#single process initialization
	processlengths = zeros(nprocsmax, dtype=int) + ns / nprocsmax
	ng_rem = ns - ns / nprocsmax * nprocsmax
	processlengths[:ng_rem] += 1
	proc_i = zeros(nprocsmax, dtype=int)
	for j in range(nprocsmax):
		proc_i[j] = sum(processlengths[:j])
	#loop for this process
	for ii in range(proc_i[proc_local],proc_i[proc_local] + processlengths[proc_local]):
		fits0 = fits.open(folders[folder]+fitslist[ii]+'/e2ds.fits')[0]
		wfile = fits0.header['HIERARCH ESO DRS CAL TH FILE'][:30] + 'wave_A.fits'
		ws = fits.open(folders[folder]+'waves/'+wfile)[0].data
		spectra0 = load(folders[folder]+fitslist[ii]+'/norm.npy') #.reshape(norder,nx)
		for j in range(norder):
			spectra0[j] = interp(ws0[j],ws[j]*(1.0+(BERVs[ii]-BORVs[ii])/2.99792458e5),spectra0[j])
		#save the interpolated spectra to files
		save(folders[folder]+fitslist[ii]+'/normRV.npy', spectra0)
		if ii==0:
			save(folders[folder]+'wave0.npy', ws0)




#make order-by-order files of time series spectra
def timeSeriesPerOrder(folder):
	os.system("rm " + folders[folder] + ".DS_Store")
	filelist = array(os.listdir(folders[folder]))
	fitslist = filelist[where(array([('20' in filelist[i][:2]) for i in range(len(filelist))]))[0]]
	fitslist = sort(fitslist)
#	fitslist = delete(fitslist, badspec[folder]) if pics0files1 else fitslist
	ns = len(fitslist)
	norder = fits.open(folders[folder]+fitslist[0]+'/e2ds.fits')[0].data.shape[0]
	if int(ceil(foldersizes[folder] / maxgb)) > norder:
		print "Warning: times series of 1 order may be too large"
	nx = fits.open(folders[folder]+fitslist[0]+'/e2ds.fits')[0].data.shape[-1]
	#single process initialization
	processlengths = zeros(nprocsmax, dtype=int) + norder / nprocsmax
	ng_rem = norder - norder / nprocsmax * nprocsmax
	processlengths[:ng_rem] += 1
	proc_i = zeros(nprocsmax, dtype=int)
	for j in range(nprocsmax):
		proc_i[j] = sum(processlengths[:j])
	totalfluxes = zeros(ns)
	totalorderfluxes = zeros((ns,norder))
	#loop for this process
	for ii in range(proc_i[proc_local],proc_i[proc_local] + processlengths[proc_local]):
		print 'Beginning flattening of order number' + str(ii)
		wave0 = zeros((ns,nx))
		spectra0 = zeros((ns,nx))
		spectra1 = zeros((ns,nx))
		spectra2 = zeros((ns,nx))
		spectra3 = zeros((ns,nx))
		for j in range(ns):
		#load individual spectra
		#file name change key: norm -> normInterp // wave0 -> wave // norm0 -> norm // normRV -> normRVInterp // onlyRV -> RVInterp OR blazeRVInterp
			ws0 = load(folders[folder]+fitslist[j]+'/wave.npy')
			sp0 = load(folders[folder]+fitslist[j]+'/norm.npy')
			sp1 = load(folders[folder]+fitslist[j]+'/normRVInterp.npy') #.reshape(norder,nx)
			sp2 = load(folders[folder]+fitslist[j]+'/RVInterp.npy') #.reshape(norder,nx)
			sp3 = load(folders[folder]+fitslist[j]+'/blazeRVInterp.npy') #.reshape(norder,nx)
			if (ii<0.5) and (ii > -0.5):
				trim = (sp2 >= percentile(sp2,10)) & (sp2 <= percentile(sp2,90))
				totalfluxes[j] = average(sp2[trim])
				for jj in range(norder):
					trim = (sp3[jj] >= percentile(sp3[jj],5)) & (sp3[jj] <= percentile(sp3[jj],95))
					totalorderfluxes[j,jj] = average(sp3[jj][trim])
			wave0[j] = ws0[ii]
			spectra0[j] = sp0[ii]
			spectra1[j] = sp1[ii]
			spectra2[j] = sp2[ii]
			spectra3[j] = sp3[ii]
		#save the stacked spectra to files
		#file name change key: unshifted_waves -> waves // unshifted_order -> norm // order -> normRVInterp // fluxes -> RVInterp OR blazeRVInterp
		save(folders[folder]+'waves' + str(ii) + '.npy', wave0)
		save(folders[folder]+'norm' + str(ii) + '.npy', spectra0)
		save(folders[folder]+'normRVInterp' + str(ii) + '.npy', spectra1)
		save(folders[folder]+'RVInterp' + str(ii) + '.npy', spectra2)
		save(folders[folder]+'blazeRVInterp' + str(ii) + '.npy', spectra3)
		if (ii<0.5) and (ii > -0.5):
			savetxt(folders[folder] + 'allspTOTALFLUX.txt', totalfluxes)
			save(folders[folder] + 'allspORDERFLUXBLAZE.npy', totalorderfluxes)


"""

#the following assumes the data for the target corresponding to datadir has been acquired and sorted using getHARPSdata.py 

#imagedir = '/lustre/scratch/aww/outputs/fitted_orders0/' if mode=='parallel' else "/Volumes/My_Passport/HARPS/AlphaCen/A/fitted_orders0/"
#imagedir = "/gpfs/group/ebf11/default/afw5465/solar_plots/"


folders = []
for i in targets:
	folders.append(datadir + i + '/')

n = len(folders)

#initialize some parameters
#maxgb = 2000000000

datefmt='%Y-%m-%dT%H:%M:%S.%f'

maskrs = {}
maskis = {}

#go back and check badspec from stitched HARPS spectra & compare to order-by-order spectra of same dates
badspec = array([[]]*n)

#extract fits data once
#if proc_local==0:
#	for folder in range(n):
#		getFitsData(folder)

#if job_type=='spec':

#make the normalized spectra files
for folder in range(n):
	normalizeSpectra(folder)
	
"""
if job_type=='ordr':
	#make the time-series per order files
	for folder in range(n):
		timeSeriesPerOrder(folder)

if job_type=='dopp':
	#make the normRV.npy files from norm.npy files
	for folder in range(n):
		dopplerShiftSpectra(folder)
"""