from numpy import *
import time
import os
import sys
from scipy.stats.stats import linregress
from astropy.io import fits
from datetime import datetime
from jdcal import gcal2jd
from tqdm import tqdm

deleteOldFiles = True

dataset = sys.argv[1]

print('Initializing...')

nprocsmax=800

if dataset=="HARPS-N_solar":
	datadir = "/gpfs/group/ebf11/default/"
	targets = ["HARPS-N_solar"]
	datasetCode="HNS"

if dataset=="AlphaCenB":
	datadir = "/gpfs/group/ebf11/default/afw5465/AlphaCen/"
	targets = ["B"]
	datasetCode="ACB"

folders = []
for i in targets:
	folders.append(datadir + i + '/')

n = len(folders)

if deleteOldFiles:
	#file name change key: norm -> normInterp // wave0 -> wave // norm0 -> norm // normRV -> normRVInterp // onlyRV -> RVInterp OR blazeRVInterp
	print('Deleting old files...')
	for folder in range(n): #delete old files
		filelist = array(os.listdir(folders[folder]))
		fitslist = filelist[where(array([('20' in filelist[i][:2]) for i in range(len(filelist))]))[0]]
		for fi in tqdm(fitslist):
			checkDir = folders[folder] + fi + "/wave.npy"
			os.system('rm ' + checkDir)
			checkDir = folders[folder] + fi + "/norm.npy"
			os.system('rm ' + checkDir)
			checkDir = folders[folder] + fi + "/normRVInterp.npy"
			os.system('rm ' + checkDir)
		print('Files deleted for folder number ' + str(folder))
print('Submitting jobs for normalization...')
for j in tqdm(range(nprocsmax)): #submit parallel jobs
	jobName = "spec" + str(j).zfill(4) + datasetCode
	qsub = "qsub -N " + jobName + " flattenHARPSorders.pbs"
	os.system(qsub)
	time.sleep(0.5)
print('Jobs submitted. Waiting for them to finish.')
for folder in range(n): #wait for the jobs to finish
	filelist = array(os.listdir(folders[folder]))
	fitslist = filelist[where(array([('20' in filelist[i][:2]) for i in range(len(filelist))]))[0]]
	for fi in fitslist:
		checkDir = folders[folder] + fi + "/wave.npy"
		while not os.path.isfile(checkDir):
			time.sleep(1)
		checkDir = folders[folder] + fi + "/norm.npy"
		while not os.path.isfile(checkDir):
			time.sleep(1)
		checkDir = folders[folder] + fi + "/normRVInterp.npy"
		while not os.path.isfile(checkDir):
			time.sleep(1)
	print('Jobs completed for folder number ' + str(folder))
print 'Done.'



"""
#file name change key: unshifted_waves -> waves // unshifted_order -> norm // order -> normRVInterp // fluxes -> RVInterp OR blazeRVInterp
norder=72
nprocsmax=norder
print('Deleting old files...')
for folder in range(n): #delete old files
	for i in range(norder):
		checkDir = folders[folder] + "waves" + str(i) + ".npy"
		os.system('rm ' + checkDir)
		checkDir = folders[folder] + "norm" + str(i) + ".npy"
		os.system('rm ' + checkDir)
		checkDir = folders[folder] + "normRVInterp" + str(i) + ".npy"
		os.system('rm ' + checkDir)
		checkDir = folders[folder] + "RVInterp" + str(i) + ".npy"
		os.system('rm ' + checkDir)
		checkDir = folders[folder] + "blazeRVInterp" + str(i) + ".npy"
		os.system('rm ' + checkDir)
	checkDir = folders[folder] + "allspTOTALFLUX.txt"
	os.system('rm ' + checkDir)
	print('Files deleted for folder number ' + str(folder))
print 'Submitting jobs for stacking orders...'
for j in range(nprocsmax): #submit parallel jobs
	jobName = "ordr" + str(j).zfill(4)
	qsub = "qsub -N " + jobName + " flattenHARPSorders.qs"
	os.system(qsub)
print 'Jobs submitted. Waiting for them to finish.'
for folder in range(n): #wait for the jobs to finish
	for i in range(norder):
		checkDir = folders[folder] + "waves" + str(i) + ".npy"
		while not os.path.isfile(checkDir):
			time.sleep(1)
		checkDir = folders[folder] + "norm" + str(i) + ".npy"
		while not os.path.isfile(checkDir):
			time.sleep(1)
		checkDir = folders[folder] + "normRVInterp" + str(i) + ".npy"
		while not os.path.isfile(checkDir):
			time.sleep(1)
		checkDir = folders[folder] + "RVInterp" + str(i) + ".npy"
		while not os.path.isfile(checkDir):
			time.sleep(1)
		checkDir = folders[folder] + "blazeRVInterp" + str(i) + ".npy"
		while not os.path.isfile(checkDir):
			time.sleep(1)
	checkDir = folders[folder] + "allspTOTALFLUX.txt"
	while not os.path.isfile(checkDir):
		time.sleep(1)
	checkDir = folders[folder] + "allspORDERFLUXBLAZE.npy"
	while not os.path.isfile(checkDir):
		time.sleep(1)
	print('Jobs completed for folder number ' + str(folder))
print 'Done.'

#compress order-by-order fits into a more readable format
qsub = "qsub combine_order_fits.qs"
os.system(qsub)
"""

"""
nprocsmax=800
print 'Submitting jobs for doppler shifting...'
for j in range(nprocsmax): #submit parallel jobs
	jobName = "dopp" + str(j).zfill(4)
	qsub = "qsub -N " + jobName + " flattenHARPSorders.qs"
	os.system(qsub)
	time.sleep(2)
print 'Jobs submitted. Waiting for them to finish.'
for folder in range(n): #wait for the jobs to finish
	filelist = array(os.listdir(folders[folder]))
	fitslist = filelist[where(array([('20' in filelist[i][:2]) for i in range(len(filelist))]))[0]]
	for fi in fitslist:
		checkDir = folders[folder] + fi + "/normRV.npy"
		while not os.path.isfile(checkDir):
			time.sleep(1)
	print 'Jobs completed for folder number ' + str(folder)
print 'Done.'
"""

