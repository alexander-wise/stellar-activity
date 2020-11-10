from numpy import *
import scipy.interpolate as interpolate
from scipy.signal import medfilt as medfilt
from scipy.optimize import curve_fit
from scipy import stats
from scipy import signal
import os
from scipy.stats.stats import linregress
from os import listdir
from scipy.stats import pearsonr
from scipy.stats import kendalltau
from astroML.time_series import lomb_scargle, lomb_scargle_BIC, lomb_scargle_bootstrap
from astropy.io import fits
from datetime import datetime
from jdcal import gcal2jd
import tarfile
from tqdm import tdqm


#First download the .fits and .tar files from the ESO archive to 'datadir' using the bash script provided (remove txt files from it using e.g. sed -i.bak '/txt/d' downloadRequest*.sh)

#Then run the following section to acquire and sort HARPS data for a single target

datadir = "/gpfs/group/ebf11/default/HARPS-N_solar/"

filelist = array(os.listdir(datadir))
fitslist = filelist[where(array([('_S2D_A.fits' in filelist[i][-11:]) for i in range(len(filelist))]))[0]]


#make folders with observation dates and move spectra files there
for fi in fitslist:
	obs = fits.open(datadir + fi)[0].header['DATE-OBS']
	os.system("mkdir " + datadir + obs)
	os.system("mv " + datadir + fi + " " + datadir + obs + "/s2d.fits")
	os.system("mv " + datadir + fi[:-11]+"_S1D_A.fits" + " " + datadir + obs + "/s1d.fits")
	os.system("mv " + datadir + fi[:-11]+"_CCF_A.fits" + " " + datadir + obs + "/ccf.fits")



"""
#extract tar files into the correct folders

filelist = array(os.listdir(datadir))
fitslist = filelist[where(array([('20' in filelist[i][:2]) for i in range(len(filelist))]))[0]]
tarlist = filelist[where(array([('tar' in filelist[i][-3:]) for i in range(len(filelist))]))[0]]

codes = ['bis', 'ccf', 'e2ds', 's1d']
for i in range(len(tarlist)):
	tar = tarfile.open(datadir + tarlist[i])
	for j in range(len(tar.getmembers())):
		try:
			obj = fits.open(tar.extractfile(tar.getmembers()[j]))[0].header['DATE-OBS']
			for k in codes:
				if k in tar.getnames()[j]:
					fname = tar.getnames()[j]
					if (fname[-6]=='A') and (not os.path.isfile(datadir+obj+'/'+k+'.fits')):
						tar.extract(tar.getmembers()[j],path=datadir+obj+'/')
						os.system('mv '+datadir+obj+'/'+fname+' '+datadir+obj+'/'+k+'.fits')
					elif (fname[-6]=='A'):
						print 'ERROR: ' + obj+'/'+k+'.fits already exists!'
					else:
						print fname + ' skipped.'
		except IOError:
			print "tar file " + str(i) + " has corrupted data in member " + str(j)




#if you want to discard some data based on time sampling, you can use the code below to visualize the distribution of observations over time
def visualizeJDs():
	filelist = array(os.listdir(datadir))
	fitslist = filelist[where(array([('20' in filelist[i][:2]) for i in range(len(filelist))]))[0]]
	datefmt='%Y-%m-%dT%H:%M:%S.%f'
	JDs = zeros(len(fitslist))
	for i in range(len(fitslist)):
		date0 = datetime.strptime(fitslist[i],datefmt)
		JDs[i] = sum(gcal2jd(date0.year,date0.month,date0.day)) + (date0.hour + (date0.minute + date0.microsecond/1e6/60.0) / 60.0) / 24.0
	keep_file_list = fitslist[argsort(JDs)]
	JDs = sort(JDs)
	JDs = JDs - 2450000
	bins = arange(floor(JDs[0]),ceil(JDs[-1]),dtype=float)
	h = hist(JDs, bins=bins)[0]
	#the code below can delete the files, it is commented out for safety
	#keep_file_list = keep_file_list[where((JDs>5200)&(JDs<6000))[0]]
	#for i in fitslist:
	#	if i not in keep_file_list:
	#		os.system('sudo rm -r ' + datadir + i)




### GET WAVELENGTH AND BLAZE CALIBRATION FILES ###


filelist = array(os.listdir(datadir))
fitslist = filelist[where(array([('20' in filelist[i][:2]) for i in range(len(filelist))]))[0]]
caldates=[]
for i in range(len(fitslist)):
	fits0 = fits.open(datadir + fitslist[i] + '/e2ds.fits')[0]
	date0 = fits0.header['HIERARCH ESO DRS CAL TH FILE'] #wavelength file
	if date0 not in caldates:
		caldates.append(date0)
	date0 = fits0.header['HIERARCH ESO DRS BLAZE FILE'] #blaze file
	if date0 not in caldates:
		caldates.append(date0)



#Use this file in the ESO "data direct retrieval" window to get the wavelength and blaze calibration files
savetxt('/Users/aww/Desktop/harpsFileList.txt',array([caldates[i][:29] for i in range(len(caldates))],dtype='object') + '.tar', fmt='%s')



def deleteAccessDeniedData():
	filelist = array(os.listdir(datadir))
	fitslist = filelist[where(array([('20' in filelist[i][:2]) for i in range(len(filelist))]))[0]]
	tarlist = filelist[where(array([('tar' in filelist[i][-3:]) for i in range(len(filelist))]))[0]]
	tarlist2 = tarlist[where(array([('HARPS' in tarlist[i][:5]) for i in range(len(tarlist))]))[0]]
	attempted_downloads = array([caldates[i][:29] for i in range(len(caldates))],dtype='object')
	successful_downloads = array([tarlist2[i][:-4] for i in range(len(tarlist2))])
	failed_downloads = []
	for i in attempted_downloads:
		if i not in successful_downloads:
			failed_downloads.append(i)
	failed_downloads = array(failed_downloads)
	for i in range(len(fitslist)):
		fits0 = fits.open(datadir + fitslist[i] + '/e2ds.fits')[0]
		date0 = fits0.header['HIERARCH ESO DRS CAL TH FILE'] #wavelength file
		if date0[:29] in failed_downloads:
			os.system('sudo rm -r ' + datadir + fitslist[i])
		date0 = fits0.header['HIERARCH ESO DRS BLAZE FILE'] #blaze file
		if date0[:29] in failed_downloads:
			os.system('sudo rm -r ' + datadir + fitslist[i])
	filelist = array(os.listdir(datadir))
	fitslist = filelist[where(array([('20' in filelist[i][:2]) for i in range(len(filelist))]))[0]]
	caldates=[]
	for i in range(len(fitslist)):
		fits0 = fits.open(datadir + fitslist[i] + '/e2ds.fits')[0]
		date0 = fits0.header['HIERARCH ESO DRS CAL TH FILE'] #wavelength file
		if date0 not in caldates:
			caldates.append(date0)
		date0 = fits0.header['HIERARCH ESO DRS BLAZE FILE'] #blaze file
		if date0 not in caldates:
			caldates.append(date0)
#note: one reason the loop below can stall indefinitely is if access is denied to some of the calibration files. If that is that case, run the function above to discard the data for which calibration files are not obtainable
keepGoing = True
while(keepGoing):
	raw_input('Use the file, harpsFileList.txt, on your Desktop in the ESO \'data direct retrieval\' window to get the wavelength calebration files. Download them into your data directory. Press enter when done.')
	keepGoing = False
	for i in range(len(caldates)):
		fname = caldates[i][:29] + '.tar' #file name if using ESO-generated download script
		fname2 = fname.replace(':','_') #file name if files downloaded manually
		if not (os.path.isfile(datadir+fname) or os.path.isfile(datadir+fname2)):
			keepGoing=True



os.system("rm " + datadir + ".DS_Store")
filelist = array(os.listdir(datadir))
fitslist = filelist[where(array([('20' in filelist[i][:2]) for i in range(len(filelist))]))[0]]
tarlist = filelist[where(array([('tar' in filelist[i][-3:]) for i in range(len(filelist))]))[0]]
tarlist1 = tarlist[where(array([('ADP' in tarlist[i][:3]) for i in range(len(tarlist))]))[0]]
tarlist2 = tarlist[where(array([('HARPS' in tarlist[i][:5]) for i in range(len(tarlist))]))[0]]


codes = ['wave', 'blaze']
labels1 = []
labels2 = []
for i in range(len(tarlist2)):
	tar = tarfile.open(datadir+tarlist2[i])
	for j in range(len(tar.getmembers())):
		for k in codes:
			if k in tar.getnames()[j]:
				labels1.append(i)
for i in range(len(tarlist2)):
	labels2.append(i)
#	if i != 74:
	labels2.append(i)

if all(array(labels1) == array(labels2)):
	print "Successfully found all wavelength and blaze calibration files!"
else:
	print "Warning: Possibly missing some wavelength or blaze calibration files!"


codes = ['wave', 'blaze']
os.system("mkdir " + datadir + "waves")
os.system("mkdir " + datadir + "blazes")
for i in range(len(tarlist2)):
	tar = tarfile.open(datadir+tarlist2[i])
	for j in range(len(tar.getmembers())):
		for k in codes:
			if k in tar.getnames()[j]:
				fname = tar.getnames()[j]
				tar.extract(tar.getmembers()[j],path=datadir+k+'s/')
				if '/' in fname:
					fname0 = fname.split('/')[-1]
					print fname0
					os.system('mv '+datadir+k+'s/'+fname+' '+datadir+k+'s/'+fname0)

"""

### EXTRACT RVs and NOISE LEVELS FROM FITS FILES ###

def getFolderSize(folder):
    total_size = os.path.getsize(folder)
    for item in os.listdir(folder):
        itempath = os.path.join(folder, item)
        if os.path.isfile(itempath):
            total_size += os.path.getsize(itempath)
        elif os.path.isdir(itempath):
            total_size += getFolderSize(itempath)
    return total_size

#produce helper files for normalizeSpectra() in flattenHARPSorders.py
def getFitsData():
	filelist = array(os.listdir(datadir))
	fitslist = filelist[where(array([('20' in filelist[i][:2]) for i in range(len(filelist))]))[0]]
	fitslist = sort(fitslist)
#	fitslist = delete(fitslist, badspec[folder]) if pics0files1 else fitslist
	ns = len(fitslist)
	#norder = fits.open(datadir+fitslist[0]+'/e2ds.fits')[0].data.shape[0]
	norder = fits.open(datadir+fitslist[0]+'/s2d.fits')[1].data.shape[0]
	#if int(ceil(foldersizes[folder] / maxgb)) > norder:
	#	print "Warning: times series of 1 order may be too large for farber/mills default memory"
	#nx = fits.open(datadir+fitslist[0]+'/e2ds.fits')[0].data.shape[-1]
	nx = fits.open(datadir+fitslist[0]+'/s2d.fits')[1].data.shape[-1]
	#get observation days
	JDs = zeros(ns)
	for j in range(ns):
		date0 = datetime.strptime(fitslist[j],datefmt)
		#JDs[j] = sum(gcal2jd(date0.year,date0.month,date0.day)) + (date0.hour + (date0.minute + date0.microsecond/1e6/60.0) / 60.0) / 24.0 #old erroneous code - some JDs may have been calculated using this formula
		JDs[j] = sum(gcal2jd(date0.year,date0.month,date0.day)) + (date0.hour + (date0.minute + (date0.second + date0.microsecond/1e6) / 60.0) / 60.0) / 24.0
	#get noise levels from (1 / signal to noise ratios)
	NLs = zeros(ns)
	#BERVs = zeros(ns) #earth barycentric RVs
	#BORVs = zeros(ns) #object barycentric RVs
	for j in tqdm(range(ns)):
		fitstemp = fits.open(datadir+fitslist[j] + '/s2d.fits')
		SNR = fitstemp[0].header['TNG QC ORDER35 SNR']
		if (SNR > 0):
			NLs[j] = 1.0 / SNR
		else:
			NLs[j] = 1000.0
	#	BERVs[j] = fitstemp[0].header['HIERARCH ESO DRS BERV']
	#	BORVs[j] = fits.open(datadir+fitslist[j] + '/ccf.fits')[0].header['HIERARCH ESO DRS CCF RVC']
	savetxt(datadir + 'allspNLs.txt', column_stack((JDs,NLs)))
	#savetxt(datadir + 'allspBRVs.txt', column_stack((JDs,BERVs,BORVs)))
"""
datadir = "/Volumes/My_Passport/HARPS/"
targets = ['eEri']
#datadir = "/Users/aww/Desktop/"
#targets = ['AlphaCenB_LowActivty']


folders = []
for i in targets:
	folders.append(datadir + i + '/')

n = len(folders)

#sort folders by size
foldersizes = zeros(n)
for i in range(n):
	foldersizes[i] = getFolderSize(folders[i])
folders = list(array(folders)[argsort(foldersizes)])
foldersizes = sort(foldersizes)
"""
#initialize some parameters
maxgb = 2000000000
datefmt='%Y-%m-%dT%H:%M:%S.%f'

getFitsData()

#If getFitsData() fails due to missing ccf.fits files, find out how many are missing:
#os.system('ls -d ' + folders[0] + '20* | wc -l')
#os.system('ls ' + folders[0] + '20*/ccf.fits | wc -l')

#if only a small number of ccf.fits files are missing:
#for j in range(ns):
#	if not os.path.isfile(folders[folder]+fitslist[j] + '/ccf.fits'):
#		os.system('rm -r ' + folders[folder]+fitslist[j])



"""
#how to find multiple targets in the ESO archive with time series data (at least 1 spectrum per 2 days for 10+ days)

#Go to Simbad, choose "Criteria query" and enter the limitations you want on your star
#maintype = 'Variable of BY Dra type' & sptypes > 'G2'
#sptypes < 'K4' & sptypes > 'G2' & Rmag < 20.0

#save the resulting list as a txt file, sim.txt, then use:
#ids = genfromtxt("Desktop/simbad.txt", delimiter='|', usecols=1, skip_header=9, skip_footer=1, dtype="S25")
#savetxt("Desktop/ids.txt",ids, fmt='%s')

#ids=genfromtxt("/Users/aww/Desktop/ids.txt",dtype='S30',delimiter='\n')

#now we have a target list file for the ESO archive
#I submitted the target list in an ESO search for HARPS reduced spectra, sorted by date

#search the list for targets with time-series observations (at least 1 observation per 2 days for 10+ days?)
datefmt='%Y-%m-%dT%H:%M:%S.%f'

obs = genfromtxt("/Users/aww/Desktop/targetlist_eso.csv",delimiter=',',skip_header=2,usecols=(1,7),dtype='S30')
obsDs = zeros(len(obs))
for i in range(len(obs)):
	obsDs[i] = datetime.toordinal(datetime.strptime(obs[i,1],datefmt))

#def findtargets(dt,mindays):
targets = []
tar0 = 'abc'
tar = 'def'
group = []
groups = []
goodtargets=[]
fullgroups=[]
ntar = 0
for i in range(len(obs)):
	tar = obs[i,0]
	if ((tar == tar0) & (abs(obsDs[i]-obsDs[i-1])<dt)):
		group.append(i-1)
	elif len(group)>0:
		targets.append(tar0)
		groups.append(group)
		group = []
	tar0=obs[i,0]
for i in range(len(targets)):
	if abs(obsDs[groups[i][-1]] - obsDs[groups[i][0]])>mindays:
		if targets[i] not in goodtargets:
			ntar+=1
		fullgroups.append(groups[i])
		goodtargets.append(targets[i])
#	return array(goodtargets), fullgroups, ntar

gt,fg,n = findtargets(4.5,40) #findtargets(4.5,50) - (0,2,4 used here) #findtargets(3.5,19)
#gt are the good targets I chose, it turns out indices 1,4,5 are the best spectra for these 3 targets
"""

#Go to ESO archive and search for spectra by target name and date range, and downloaded the .fits and .tar files using the bash script provided (remove txt files from it using e.g. sed -i.bak '/txt/d' downloadRequest*.sh)



"""
#rename files to observation dates
for i in folders:
	os.system("rm " + i + ".DS_Store")
for i in range(len(folders)):
	filelist = listdir(folders[i])
	for fi in filelist:
		os.system("mv " + folders[i] + fi + " " + folders[i] + fits.open(folders[i] + fi)[0].header['DATE-OBS'])
"""

"""
#split alpha cen A and B, fits and tar files into their respective folders


filelist = array(os.listdir("/Volumes/My_Passport"))

fitslist = filelist[where(array([('20' in filelist[i][:2]) for i in range(len(filelist))]))[0]]

tarlist = filelist[where(array([('tar' in filelist[i][-5:]) for i in range(len(filelist))]))[0]]

tarlist1 = tarlist[where(array([('ADP' in tarlist[i][:3]) for i in range(len(tarlist))]))[0]]
tarlist2 = tarlist[where(array([('HARPS' in tarlist[i][:5]) for i in range(len(tarlist))]))[0]]


for i in range(len(fitslist)):
	xfits = fits.open(fitslist[i])[0].header['OBJECT']
	if xfits=='HD128621':
		os.system('mv ' + fitslist[i] + ' B/')
	else:
		os.system('mv ' + fitslist[i] + ' A/')

for i in range(len(tarlist)):
	tar = tarfile.open(tarlist[i])
	j=0
	obj = 'aaa'
	while (obj[:2] != 'HD') and (j < len(tar.getmembers())):
		obj = fits.open(tar.extractfile(tar.getmembers()[j]))[0].header['OBJECT']
		j+=1
	print obj
	if obj=='HD128621':
		os.system('mv ' + tarlist[i] + ' ../B/')
	else:
		os.system('mv ' + tarlist[i] + ' A/')

dates = []
for i in range(len(tarlist)):
	tar = tarfile.open(tarlist[i])
	j=0
	obj = 'aaa'
	while (obj[:2] != '20') and (j < len(tar.getmembers())):
		obj = fits.open(tar.extractfile(tar.getmembers()[j]))[0].header['DATE-OBS']
		j+=1
	dates.append(obj)
	if obj=='HD128621':
		os.system('mv ' + tarlist[i] + ' ../B/')
	else:
		os.system('mv ' + tarlist[i] + ' A/')



for i in folders:
	os.system("rm " + i + ".DS_Store")
for i in range(len(folders)):
	for fi in fitslist:
		os.system("mkdir " + fi + "xx")
		os.system("mv " + fi + " " + fi + 'xx/fullsp.fits')
		os.system("mv " + fi + "xx " + fi)


codes = ['bis', 'ccf', 'e2ds', 's1d']
for i in range(len(tarlist)):
	tar = tarfile.open(tarlist[i])
	for j in range(len(tar.getmembers())):
		try:
			obj = fits.open(tar.extractfile(tar.getmembers()[j]))[0].header['DATE-OBS']
			for k in codes:
				if k in tar.getnames()[j]:
					fname = tar.getnames()[j]
					if (fname[-6]=='A') and (not os.path.isfile(obj+'/'+k+'.fits')):
						tar.extract(tar.getmembers()[j],path=obj+'/')
						os.system('mv '+obj+'/'+fname+' '+obj+'/'+k+'.fits')
					elif (fname[-6]=='A'):
						print 'ERROR: ' + obj+'/'+k+'.fits already exists!'
					else:
						print fname + ' skipped.'
		except IOError:
			print "tar file " + str(i) + " has corrupted data in member " + str(j)



codes = ['wave']
labels = []
for i in range(len(tarlist2)):
	tar = tarfile.open(tarlist2[i])
	for j in range(len(tar.getmembers())):
		for k in codes:
			if k in tar.getnames()[j]:
				labels.append(i)
#				print fits.open(tar.extractfile(tar.getmembers()[j]))[0].header['HIERARCH ESO DRS CAL TH FILE']

dates=[]
for i in range(len(fitslist)):
	date0= fits.open(fitslist[i] + '/e2ds.fits')[0].header['HIERARCH ESO DRS CAL TH FILE']
	if date0 not in dates:
		dates.append(date0)

"""

