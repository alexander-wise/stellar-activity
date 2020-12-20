# transform AlphaCenB_normalized folder from my google drive into several test data sets

import sys
from numpy import *
import scipy.interpolate as interpolate
from tqdm import tqdm

#input_1 = sys.argv[1]
#input_2 = sys.argv[2]


RVdataDir = '/Users/aww/Desktop/RV_plots/'

#loaddir = '/storage/work/afw5465/AlphaCen/testData/'
#RVdataDir = '/storage/work/afw5465/AlphaCen/'


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




JDs = load(RVdataDir+'JDs.npy')
BERVs = load(RVdataDir+'BERVs.npy')
BORVs = load(RVdataDir+'BORVs.npy')
binaryRV = load(RVdataDir+'binaryRV.npy')
#plot(JDs,BORVs-binaryRV,'.')

"""
for i in tqdm(range(72)):
	ws = genfromtxt(loaddir+'waves'+str(i)+'.txt')
	sp = genfromtxt(loaddir+'norm'+str(i)+'.txt')
	for j in range(2491):
		with open(savedir+'waves'+str(j)+'.txt', ('wb' if i==0 else 'ab')) as myfile:
			savetxt(myfile, ws[j], fmt='%.10f', delimiter=',', newline=' ')
			if i != 72-1:
				myfile.write(b'\n')
		with open(savedir+'norm'+str(j)+'.txt', ('wb' if i==0 else 'ab')) as myfile:
			savetxt(myfile, sp[j], fmt='%.10f', delimiter=',', newline=' ')
			if i != 72-1:
				myfile.write(b'\n')
"""
"""
nf=2491
nf100 = nf/100+1

BERVmidi = argmin(abs(BERVs-((amin(BERVs) + amax(BERVs))/2.)))
waves_mid = genfromtxt(loaddir+'waves'+str(BERVmidi)+'.txt')


loaddir = '/Volumes/SD/AlphaCen/testData/'


if input_1==1:
	savedir = '/storage/work/afw5465/AlphaCen/not_shifted/'
	for i in range(input_2*nf100,min((input_2+1)*nf100,nf)):
		ws0 = genfromtxt(loaddir+'waves'+str(i)+'.txt')
		sp0 = genfromtxt(loaddir+'norm'+str(i)+'.txt')
		sp1 = zeros(sp0.shape)
		for o in range(72):
			sinc_interp = BandLimitedInterpolator()
			sp1[o] = sinc_interp.interpolate(waves_mid[o],ws0[o],sp0[o])
		ws, sp = ordersTo1D(waves_mid,sp1)
		savetxt(savedir+'waves'+str(i)+'.txt', ws, fmt='%.10f')
		savetxt(savedir+'norm'+str(i)+'.txt', sp, fmt='%.10f')


if input_1==2:
	savedir = '/storage/work/afw5465/AlphaCen/BERV_shifted/'
	for i in range(input_2*nf100,min((input_2+1)*nf100,nf)):
		ws0 = genfromtxt(loaddir+'waves'+str(i)+'.txt')
		sp0 = genfromtxt(loaddir+'norm'+str(i)+'.txt')
		sp1 = zeros(sp0.shape)
		for o in range(72):
			sinc_interp = BandLimitedInterpolator()
			sp1[o] = sinc_interp.interpolate(waves_mid[o],ws0[o]*(1.0+BERVs[i]/2.99792458e8),sp0[o])
		ws, sp = ordersTo1D(waves_mid,sp1)
		savetxt(savedir+'waves'+str(i)+'.txt', ws, fmt='%.10f')
		savetxt(savedir+'norm'+str(i)+'.txt', sp, fmt='%.10f')


if input_1==3:
	savedir = '/storage/work/afw5465/AlphaCen/BERV-Binary_shifted/'
	for i in range(input_2*nf100,min((input_2+1)*nf100,nf)):
		ws0 = genfromtxt(loaddir+'waves'+str(i)+'.txt')
		sp0 = genfromtxt(loaddir+'norm'+str(i)+'.txt')
		sp1 = zeros(sp0.shape)
		for o in range(72):
			sinc_interp = BandLimitedInterpolator()
			sp1[o] = sinc_interp.interpolate(waves_mid[o],ws0[o]*(1.0+(BERVs[i]-binaryRV[i])/2.99792458e8),sp0[o])
		ws, sp = ordersTo1D(waves_mid,sp1)
		savetxt(savedir+'waves'+str(i)+'.txt', ws, fmt='%.10f')
		savetxt(savedir+'norm'+str(i)+'.txt', sp, fmt='%.10f')


if input_1==4:
	savedir = '/storage/work/afw5465/AlphaCen/BERV-BORV_shifted/'
	for i in range(input_2*nf100,min((input_2+1)*nf100,nf)):
		ws0 = genfromtxt(loaddir+'waves'+str(i)+'.txt')
		sp0 = genfromtxt(loaddir+'norm'+str(i)+'.txt')
		sp1 = zeros(sp0.shape)
		for o in range(72):
			sinc_interp = BandLimitedInterpolator()
			sp1[o] = sinc_interp.interpolate(waves_mid[o],ws0[o]*(1.0+(BERVs[i]-BORVs[i])/2.99792458e8),sp0[o])
		ws, sp = ordersTo1D(waves_mid,sp1)
		savetxt(savedir+'waves'+str(i)+'.txt', ws, fmt='%.10f')
		savetxt(savedir+'norm'+str(i)+'.txt', sp, fmt='%.10f')
"""
"""
#to make the master spectrum, take the first 250 BERV_shifted 1-D spectra and sum them. Then, to construct each "zero noise" test data set, apply the BERV, BERV-Binary, BERV-BORV, and 0 RV shifts to the master spectrum wavelengths and re-interpolate the shifted versions of the master spectrum back onto a single wavelength grid

loaddir = '/Volumes/My_Passport/HARPS/AlphaCen/BERV-BORV_shifted/'

ws0 = genfromtxt(loaddir+'waves0.txt')
master_sp = zeros(ws0.shape)

for i in tqdm(range(250)):
	sp0 = genfromtxt(loaddir+'norm'+str(i)+'.txt')
	master_sp += sp0
master_sp /= 250.


loaddir = '/Volumes/My_Passport/HARPS/AlphaCen/not_shifted/'
RVs = BORVs-BERVs
for i in tqdm(range(2491)):
	sp = interp(ws0, ws0 * (1.0+RVs[i]/2.99792458e8), master_sp)
	savetxt(loaddir+'master'+str(i)+'.txt', sp, fmt='%.10f')
savetxt(loaddir+'JDs.txt', JDs, fmt='%.8f')
savetxt(loaddir+'RVs.txt', RVs, fmt='%.10f')


loaddir = '/Volumes/My_Passport/HARPS/AlphaCen/BERV_shifted/'
RVs = BORVs
for i in tqdm(range(2491)):
	sp = interp(ws0, ws0 * (1.0+RVs[i]/2.99792458e8), master_sp)
	savetxt(loaddir+'master'+str(i)+'.txt', sp, fmt='%.10f')
savetxt(loaddir+'JDs.txt', JDs, fmt='%.8f')
savetxt(loaddir+'RVs.txt', RVs, fmt='%.10f')


loaddir = '/Volumes/My_Passport/HARPS/AlphaCen/BERV-Binary_shifted/'
RVs = BORVs-binaryRV
for i in tqdm(range(2491)):
	sp = interp(ws0, ws0 * (1.0+RVs[i]/2.99792458e8), master_sp)
	savetxt(loaddir+'master'+str(i)+'.txt', sp, fmt='%.10f')
savetxt(loaddir+'JDs.txt', JDs, fmt='%.8f')
savetxt(loaddir+'RVs.txt', RVs, fmt='%.10f')


loaddir = '/Volumes/My_Passport/HARPS/AlphaCen/BERV-BORV_shifted/'
RVs = zeros(len(JDs))
for i in tqdm(range(2491)):
	sp = interp(ws0, ws0 * (1.0+RVs[i]/2.99792458e8), master_sp)
	savetxt(loaddir+'master'+str(i)+'.txt', sp, fmt='%.10f')
savetxt(loaddir+'JDs.txt', JDs, fmt='%.8f')
savetxt(loaddir+'RVs.txt', RVs, fmt='%.10f')
"""

"""
#cut the spectra to exlude wavelengths below 4000
loaddir = '/Volumes/My_Passport/HARPS/AlphaCen/no_RV/'
ws = genfromtxt(loaddir+'interp_wavelengths.txt')
w_i = where(ws>4000)[0][0]
lw = len(ws)
for i in tqdm(range(2491)):
	sp = genfromtxt(loaddir+'norm'+str(i)+'.txt')
	savetxt(loaddir+'norm'+str(i)+'.txt', sp[-(lw-w_i):], fmt='%.10f')
	sp = genfromtxt(loaddir+'master'+str(i)+'.txt')
	savetxt(loaddir+'master'+str(i)+'.txt', sp[-(lw-w_i):], fmt='%.10f')
savetxt(loaddir+'interp_wavelengths.txt', ws[w_i:])


loaddir = '/Volumes/My_Passport/HARPS/AlphaCen/stellar_RV/'
ws = genfromtxt(loaddir+'interp_wavelengths.txt')
w_i = where(ws>4000)[0][0]
lw = len(ws)
for i in tqdm(range(2491)):
	sp = genfromtxt(loaddir+'norm'+str(i)+'.txt')
	savetxt(loaddir+'norm'+str(i)+'.txt', sp[-(lw-w_i):], fmt='%.10f')
	sp = genfromtxt(loaddir+'master'+str(i)+'.txt')
	savetxt(loaddir+'master'+str(i)+'.txt', sp[-(lw-w_i):], fmt='%.10f')
savetxt(loaddir+'interp_wavelengths.txt', ws[w_i:])


loaddir = '/Volumes/My_Passport/HARPS/AlphaCen/stellar+binary_RV/'
ws = genfromtxt(loaddir+'interp_wavelengths.txt')
w_i = where(ws>4000)[0][0]
lw = len(ws)
for i in tqdm(range(2491)):
	sp = genfromtxt(loaddir+'norm'+str(i)+'.txt')
	savetxt(loaddir+'norm'+str(i)+'.txt', sp[-(lw-w_i):], fmt='%.10f')
	sp = genfromtxt(loaddir+'master'+str(i)+'.txt')
	savetxt(loaddir+'master'+str(i)+'.txt', sp[-(lw-w_i):], fmt='%.10f')
savetxt(loaddir+'interp_wavelengths.txt', ws[w_i:])


loaddir = '/Volumes/My_Passport/HARPS/AlphaCen/stellar+binary+earth_RV/'
ws = genfromtxt(loaddir+'interp_wavelengths.txt')
w_i = where(ws>4000)[0][0]
lw = len(ws)
for i in tqdm(range(2491)):
	sp = genfromtxt(loaddir+'norm'+str(i)+'.txt')
	savetxt(loaddir+'norm'+str(i)+'.txt', sp[-(lw-w_i):], fmt='%.10f')
	sp = genfromtxt(loaddir+'master'+str(i)+'.txt')
	savetxt(loaddir+'master'+str(i)+'.txt', sp[-(lw-w_i):], fmt='%.10f')
savetxt(loaddir+'interp_wavelengths.txt', ws[w_i:])
"""


#add 3 different planets to the stellar_RV spectra: peak-to-peak amplitudes are 3, 10, and 40 m/s. phases are 1.2, 2.7, and 4.7 radians.
loaddir = '/Volumes/My_Passport/HARPS/AlphaCen/stellar_RV/'

savedir = '/Volumes/My_Passport/HARPS/AlphaCen/stellar+planet1_RV/'
ws = genfromtxt(loaddir+'interp_wavelengths.txt')
RVs = genfromtxt(loaddir+'RVs.txt') + 1.5*sin((JDs-2455000)*2.*pi/10.+1.2)
savetxt(savedir+'RVs.txt', RVs, fmt='%.10f')
for i in tqdm(range(2491)):
	sp = genfromtxt(loaddir+'norm'+str(i)+'.txt')
	savetxt(savedir+'norm'+str(i)+'.txt', interp(ws, ws * (1.0+RVs[i]/2.99792458e8), sp), fmt='%.10f')
	sp = genfromtxt(loaddir+'master'+str(i)+'.txt')
	savetxt(savedir+'master'+str(i)+'.txt', interp(ws, ws * (1.0+RVs[i]/2.99792458e8), sp), fmt='%.10f')


savedir = '/Volumes/My_Passport/HARPS/AlphaCen/stellar+planet2_RV/'
ws = genfromtxt(loaddir+'interp_wavelengths.txt')
RVs = genfromtxt(loaddir+'RVs.txt') + 5.0*sin((JDs-2455000)*2.*pi/10.+2.7)
savetxt(savedir+'RVs.txt', RVs, fmt='%.10f')
for i in tqdm(range(2491)):
	sp = genfromtxt(loaddir+'norm'+str(i)+'.txt')
	savetxt(savedir+'norm'+str(i)+'.txt', interp(ws, ws * (1.0+RVs[i]/2.99792458e8), sp), fmt='%.10f')
	sp = genfromtxt(loaddir+'master'+str(i)+'.txt')
	savetxt(savedir+'master'+str(i)+'.txt', interp(ws, ws * (1.0+RVs[i]/2.99792458e8), sp), fmt='%.10f')


savedir = '/Volumes/My_Passport/HARPS/AlphaCen/stellar+planet3_RV/'
ws = genfromtxt(loaddir+'interp_wavelengths.txt')
RVs = genfromtxt(loaddir+'RVs.txt') + 20.0*sin((JDs-2455000)*2.*pi/10.+4.7)
savetxt(savedir+'RVs.txt', RVs, fmt='%.10f')
for i in tqdm(range(2491)):
	sp = genfromtxt(loaddir+'norm'+str(i)+'.txt')
	savetxt(savedir+'norm'+str(i)+'.txt', interp(ws, ws * (1.0+RVs[i]/2.99792458e8), sp), fmt='%.10f')
	sp = genfromtxt(loaddir+'master'+str(i)+'.txt')
	savetxt(savedir+'master'+str(i)+'.txt', interp(ws, ws * (1.0+RVs[i]/2.99792458e8), sp), fmt='%.10f')

