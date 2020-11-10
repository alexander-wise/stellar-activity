#find spot sensitive lines

from numpy import *
import os
import sys
from scipy.stats.stats import linregress
from astropy.io import fits
from datetime import datetime
from jdcal import gcal2jd
from tqdm import tqdm

mode = 'parallel'
#mode = 'home'

#read in the input parameters
job_type = sys.argv[1] if mode=='parallel' else 'norm' #spec:normalization or ordr:stacking for parallel jobs
proc_local = int(sys.argv[2]) if mode=='parallel' else 0 #local process index
nprocsmax=72 if job_type=='ordr' else 1 if job_type=='norm' else 800


#datadir = "/lustre/work/phys/aww/spectra/AlphaCen/"  if mode=='parallel' else "/Volumes/My_Passport/HARPS/AlphaCen/"
#datadir = "/lustre/work/phys/aww/spectra/HARPS/"  if mode=='parallel' else "/Volumes/My_Passport/HARPS/"
datadir = "/gpfs/group/ebf11/default/"
targets = ["HARPS-N_solar"]
#targets = ['A']
#targets = ['EpsEri']


#datadir = "/Users/aww/Desktop/"
#targets = ['tau_ceti_spectra', 'fiber_calibration_spectra']


print 'nprocsmax=' + str(nprocsmax)

print 'Beginning run number ' + str(proc_local)

def xn(x0,p0):
	return poly1d(p0)(x0)
"""
def getFolderSize(folder):
    total_size = os.path.getsize(folder)
    for item in os.listdir(folder):
        itempath = os.path.join(folder, item)
        if os.path.isfile(itempath):
            total_size += os.path.getsize(itempath)
        elif os.path.isdir(itempath):
            total_size += getFolderSize(itempath)
    return total_size

#quick function - can be run locally on laptop to produce helper files for normalizeSpectra()
def getFitsData(folder):
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
	#get observation days
	JDs = zeros(ns)
	for j in range(ns):
		date0 = datetime.strptime(fitslist[j],datefmt)
		JDs[j] = sum(gcal2jd(date0.year,date0.month,date0.day)) + (date0.hour + (date0.minute + date0.microsecond/1e6/60.0) / 60.0) / 24.0
	#get noise levels from (1 / signal to noise ratios)
	NLs = zeros(ns)
	BERVs = zeros(ns) #earth barycentric RVs
	BORVs = zeros(ns) #object barycentric RVs
	for j in range(ns):
		fitstemp = fits.open(folders[folder]+fitslist[j] + '/fullsp.fits')
		SNR = fitstemp[0].header['SNR']
		if (SNR > 0):
			NLs[j] = 1.0 / SNR
		else:
			NLs[j] = 1000.0
		BERVs[j] = fitstemp[0].header['HIERARCH ESO DRS BERV']
		BORVs[j] = fits.open(folders[folder]+fitslist[j] + '/ccf.fits')[0].header['HIERARCH ESO DRS CCF RVC']
	savetxt(folders[folder] + 'allspNLs.txt', column_stack((JDs,NLs)))
	savetxt(folders[folder] + 'allspBRVs.txt', column_stack((JDs,BERVs,BORVs)))
"""
#normalize the spectra and make pictures or normalized data files
def normalizeSpectra(folder):
	#os.system("rm " + folders[folder] + ".DS_Store")
	filelist = array(os.listdir(folders[folder]))
	fitslist = filelist[where(array([('20' in filelist[i][:2]) for i in range(len(filelist))]))[0]]
	fitslist = sort(fitslist)
	#fitslist = delete(fitslist, badspec[folder]) if pics0files1 else fitslist
	ns = len(fitslist)
	norder = fits.open(folders[folder]+fitslist[0]+'/s2d.fits')['SCIDATA'].data.shape[0]
	#if int(ceil(foldersizes[folder] / maxgb)) > norder:
	#	print "Warning: times series of 1 order may be too large (this warning only valid before ANY reduced spectra files have been added to data directory)"
	nx = fits.open(folders[folder]+fitslist[0]+'/s2d.fits')['SCIDATA'].data.shape[-1]
	#get observation days
	JDs = genfromtxt(folders[folder] + 'allspNLs.txt')[:,0]
	#get radial velocities
	#BERVs = genfromtxt(folders[folder] + 'allspBRVs.txt')[:,1]
	#BORVs = genfromtxt(folders[folder] + 'allspBRVs.txt')[:,2]
	#use the wavelength scale of the last spectrum as my RV-interpolation domain
	ws0 = fits.open(folders[folder]+fitslist[-1]+'/s2d.fits')['WAVEDATA_AIR_BARY'].data
	#ws0 = fits.open(folders[folder]+'waves/'+wfile0)[0].data
	#find the normalization functions
	#blaze = fits.open(datadir + "HARPS.2009-06-01T20_00_04.000_blaze_A.fits")[0].data
	nbox = 100
	nperbox=nx/nbox+1
	p = 2
	tolerance=zeros(norder) + 0.005 #use these values for initializing tolerance determination
	tolerance[:14] += array([ 0.035,  0.0,  0.075,  0.035,  0.015,  0.0,  0.005,  0.0,
        0.0,  0.005,  0.0,  0.0,  0.0,  0.005])
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
		fits0 = fits.open(folders[folder]+fitslist[ii]+'/s2d.fits')
		#wfile = fits0.header['HIERARCH ESO DRS CAL TH FILE'][:30] + 'wave_A.fits'
		#print 'wfile=' + wfile
		ws = fits0['WAVEDATA_AIR_BARY'].data
		#bfile = fits0.header['HIERARCH ESO DRS BLAZE FILE'] #[:30] + 'blaze_A.fits'
		#print 'bfile=' + bfile
		#blaze = fits.open(folders[folder]+'blazes/'+bfile)[0].data
		spectra0 = fits0['SCIDATA'].data # / blaze #spectra to be normalized
		#fits0 = fits.open(folders[folder]+fitslist[ii]+'/e2ds.fits')[0] #reset this because otherwise modifying spectra1 will modify spectra0
		spectra1 = fits0['SCIDATA'].data #spectra saved for later RV shifting - unnormalized spectra
		spectra1[where(spectra1<0)]=0.0
		spectraErr = fits0['ERRDATA'].data
		for j in range(norder):
			if len(where(isnan(spectra0[j]))[0]) > 0.5:
				fitFailed[j] = 1.0
			else:
				spectraWeights = spectraErr[j] #sqrt(spectra1[j]) / sum(sqrt(spectra1[j])) #sqrt because numpy.polyfit says weights should be 1/sigma
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
	#			r0 = linregress(y1,xn(x1,ps[j]))[2] #unweighted linear regression r-value
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
		#							r1[k] = linregress(y1,xn(x1,p1[k]))[2] #unweighted linear regression r-value
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
		"""
	#if not pics0files1:
		#save the fits as plots
	#	matplotlib.interactive(False)
		for j in range(norder):
			savetxt(imagedir + "n" + str(folder) + "o" + str(j) + "s" + str(ii) + "S.txt", column_stack((ws[j],spectra0[j]))) #S stands for spectrum
	#		clf() #
	#		j+=1 #
	#		plot(ws[j],spectra0[j]) #
			if (folder,ii,j) in maskis:
				xs0 = delete(xs[j],maskis[(folder,ii,j)])
				ys0 = delete(ys[j],maskis[(folder,ii,j)])
			else:
				xs0 = xs[j]
				ys0 = ys[j]
	#		plot(xs0,ys0, 'k.', ms=10) #
	#		plot(xs[j],xn(xs[j],ps[j])) #
	#		print(j) #
			savetxt(imagedir + "n" + str(folder) + "o" + str(j) + "s" + str(ii) + "M.txt", column_stack((xs0,ys0))) #M stands for masked data used in linear fit
			savetxt(imagedir + "n" + str(folder) + "o" + str(j) + "s" + str(ii) + "F.txt", column_stack((xs[j],xn(xs[j],ps[j])))) #F stands for fitting function
	#	matplotlib.interactive(True)
	#else:
		"""
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
			for j in range(norder):
				spectra2[j] = interp(ws0[j],ws[j],spectra0[j])
			save(folders[folder]+fitslist[ii]+'/normInterp.npy', spectra2)
			save(folders[folder]+fitslist[ii]+'/wave.npy', ws)
			save(folders[folder]+fitslist[ii]+'/norm.npy', spectra0)
			if sum(fitFailed) > 0.5:
				savetxt(folders[folder]+fitslist[ii]+'/failedFits.txt', where(fitFailed>0.5)[0], fmt='%i')
			#spectra3 = zeros(spectra0.shape)
			#spectra4 = zeros(spectra0.shape)
			#RV correct the spectra and interpolate all of them onto a single wavelength domain
			#for j in range(norder):
			#	spectra0[j] = interp(ws0[j],ws[j]*(1.0+(BERVs[ii]-BORVs[ii])/2.99792458e5),spectra0[j])
				#spectra3[j] = interp(ws0[j],ws[j]*(1.0+(BERVs[ii]-BORVs[ii])/2.99792458e5),spectra1[j])
				#spectra4[j] = interp(ws0[j],ws[j]*(1.0+(BERVs[ii]-BORVs[ii])/2.99792458e5),spectra1[j] / blaze[j])
			#save the RV corrected and interpolated spectra to files
			#save(folders[folder]+fitslist[ii]+'/normRVInterp.npy', spectra0) #normalized, RV shifted, and interpolated
			#save(folders[folder]+fitslist[ii]+'/RVInterp.npy', spectra3) #RV shifted and interpolated
			#save(folders[folder]+fitslist[ii]+'/blazeRVInterp.npy', spectra4) #RV shifted, interpolated, and blaze shifted
			if ii==0:
				save(folders[folder]+'wave0.npy', ws0)



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

"""
#sort folders by size
foldersizes = zeros(n)
for i in range(n):
	foldersizes[i] = getFolderSize(folders[i])
folders = list(array(folders)[argsort(foldersizes)])
foldersizes = sort(foldersizes)
"""

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