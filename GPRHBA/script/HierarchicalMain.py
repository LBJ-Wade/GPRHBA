"""
Main script of the population inference
"""
import numpy as np
import argparse
from InterpUnit import InterpUnit

"""
Arguments which are parsed to this scripts.
For the posterior file and simulation file format,
please follow the format given in the example and by the wiki.
"""

parser = argparse.ArgumentParser(description='Hierarchial Bayesian inference with Gaussian process regession to interpolate the hyper parameter space.')
parser.add_argument('--PosteriorFile',type=str,help='Posterior file name',required=True)
parser.add_argument('--SimulationFile',type=str,help='Simulation file name',required=True)
parser.add_argument('--OutputDirectory',type=str,help='Output directory name',required=True)
parser.add_argument('--tag',type=str,help='Tag of the file',default='test')
parser.add_argument('--npoints',type=int,help='Number of live points',default=10)
parser.add_argument('--PCAtolerance',type=float,help='PCAtolerance',default=0.1)
parser.add_argument('--EvidenceTolerance',type=float,help='EvidenceTolerance',default=0.5)
parser.add_argument('--Approach',type=str,help='Approach for calculating the likelihood',choices=['BottomUp','TopDown'],default='TopDown')
parser.add_argument('--WithRate',type=str,help='Using rate information or not.',default='True')

args = parser.parse_args()
posteriorFile = args.PosteriorFile
simulationFile = args.SimulationFile
outputDirectory = args.OutputDirectory
tag = args.tag
npoints = args.npoints
PCAtolerance = args.PCAtolerance
evidenceTolerance = args.EvidenceTolerance
Approach = args.Approach
WithRate = eval(args.WithRate)

# ======================================================
# Load GW posterior and prior
# ======================================================

"""
The shape of the postierior should be (nEvent,nSample,nDim).
nEvent is the number of events in the catalog.
nSample is the number of posterior sample for that particular event.
nDim is the number of dimension in the waveform parameter.
"""

data = np.load(posteriorFile)
posterior_vector = data['posterior_vector']
prior_vector = data['prior_vector']
length_array = data['length_array']
Tobs = data['Tobs']

# ======================================================
# Load training data and construct Population Likelihood
# ======================================================
"""
This part of the code is fairly general for input that fit the input format,
DO NOT modify this part unless you find a critical bug or you fully
understand this part.
"""

simulations = np.load(simulationFile)
histBins = simulations['histBins']
Observed = simulations['Observed']*Tobs
Intrinsic = simulations['Intrinsic']*Tobs
hyperParameterDesign = simulations['hyperParameterDesign']
nSim = Intrinsic.shape[0]

Observed = np.log10(np.array(Observed).T)
Observed[np.isinf(Observed)] = -20
Intrinsic = np.log10(np.array(Intrinsic).T)
Intrinsic[np.isinf(Intrinsic)] = -20
norm_input = np.log10((np.sum(np.power(10,Intrinsic),axis=0)/np.sum(np.power(10,Observed),axis=0))).reshape(1,nSim)

Interp_norm = InterpUnit(norm_input,histBins,np.array([hyperParameterDesign]).T,tol=PCAtolerance)
Interp_intrinsic = InterpUnit(Intrinsic,histBins,np.array([hyperParameterDesign]).T,tol=PCAtolerance)
Interp_observed = InterpUnit(Observed,histBins,np.array([hyperParameterDesign]).T,tol=PCAtolerance)

if WithRate == True:
	if Approach =='BottomUp':	
		def LogLikelihood(cube):
			if type(cube) is not np.ndarray:
				cube = np.array([cube])
			p_theta = Interp_intrinsic.GetPtheta(posterior_vector,cube)
			normalization = Interp_norm.GetNormalization(cube)
			rate = Interp_observed.GetNormalization(cube)
			single_mean = np.zeros(length_array.size-1)
			for i in range(length_array.size-1):
			    single_mean[i] = np.mean(p_theta[length_array[i]:length_array[i+1]]/prior_vector[i])*normalization
			return np.sum(np.log(single_mean[np.nonzero(single_mean)]))-rate+(length_array.size-1)*np.log(rate)
	else:
		def LogLikelihood(cube):
			if type(cube) is not np.ndarray:
				cube = np.array([cube])
			p_theta = Interp_intrinsic.GetNtheta(posterior_vector,cube)
			normalization = Interp_norm.GetNormalization(cube)
			rate = Interp_observed.GetNormalization(cube)
			single_mean = np.zeros(length_array.size-1)
			for i in range(length_array.size-1):
			    single_mean[i] = np.mean(p_theta[length_array[i]:length_array[i+1]]/prior_vector[length_array[i]:length_array[i+1]])
			return np.sum(np.log(single_mean[np.nonzero(single_mean)]))-rate
else:
	def LogLikelihood(cube):
		if type(cube) is not np.ndarray:
			cube = np.array([cube])
		p_theta = Interp_intrinsic.GetPtheta(posterior_vector,cube)
		normalization = Interp_norm.GetNormalization(cube)
		single_mean = np.zeros(length_array.size-1)
		for i in range(length_array.size-1):
		    single_mean[i] = np.mean(p_theta[length_array[i]:length_array[i+1]]/prior_vector[i])*normalization
		return np.sum(np.log(single_mean[np.nonzero(single_mean)]))

# ======================================================
# Calling Inference packages
# ======================================================

"""
Once we finish constucting the population likelihood through the GPR,
we proceed to contruct the posterior function and sample it.
We use EMCEE (https://emcee.readthedocs.io/en/stable/)
as our sampler so the following section are written to 
fit the EMCEE sampling convention.
Please customize this part of the code to fit your purpose.
"""
import emcee
from multiprocessing import Pool

def LogPrior(cube):
	"""Log of population prior.

	This function defines the shape of the prior. Please see emcee
	documentation to check how prior are define.

	Args:
		cube: Coordinates in the hyper-parameter space.

	Return:
		(Float): the prior density at the coordinates in the
		hyper-parameter space.	

	"""
	if (cube<0)+(cube>265):
		return -np.inf
	else:
		return np.log10(1./265)

LogLikelihood = np.vectorize(LogLikelihood)
LogPrior = np.vectorize(LogPrior)

def LogProb(cube):
# Adding the log likelihood and log prior to get the log posterior.
	return LogLikelihood(cube)+LogPrior(cube)


nwalkers,ndim = 120,1
p0 = np.array([np.random.uniform(0,265,nwalkers)]).T
pool = Pool(24)

sampler = emcee.EnsembleSampler(nwalkers,ndim,LogProb,pool=pool)
p0,_,_ = sampler.run_mcmc(p0,100) # Burn-in steps
sampler.reset()
p0,_,_ = sampler.run_mcmc(p0,npoints) # Sampling and output
np.savez(outputDirectory,samples=sampler.flatchain)

