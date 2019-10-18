import numpy as np
import dataCompress
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel
from scipy import stats
from scipy.interpolate import griddata,interpn

class InterpUnit():
	"""
	Class which hold the data compress and gaussian process regression unit.
	
	Args:
		dataMat: (N,M) array of the histogram. N is the size of the flatten histogram, M is the number of simulations.
		histBins: (N,D) coordinate of the center of bins. D is the dimension of the histogram.
		simDesign: (M,d) coordinate in the hyperparameter space. d is the dimension in hyperparameter space.
		tol(Optional):	Tolerance in PCA.

	Attribute:
		binAxes: Axes which were used to constructed the multidimensional histogram grid point.
		binShape: Length of each axes of multidimensional histogram.
		ndim: Number of dimension of the histogram.
		gp: Array holding all the GP object.
	
	---------
	
	
	"""
	def __init__(self,dataMat,histBins,simDesign,tol=1e-5):
		self.dataMat = dataMat
		self.histBins = histBins
		self.binAxes = []
		self.binShape = []
		for i in range(histBins.shape[1]):
			self.binAxes.append(np.unique(histBins.T[i]))
			self.binShape.append(self.binAxes[i].size)
		self.binAxes = np.array(self.binAxes)
		self.binShape = np.array(self.binShape)
		self.simDesign = simDesign
		self.tol = tol
		self.ndim = simDesign.T.shape[0]
		self.InitializeDatComp()
		self.InitializeGaussianProcess()
	
	def InitializeDatComp(self):
		"""
		Initialization function for PCA compression.
		"""
		print('Initializing data compression unit.')
		print('The PCA compression tolerance is '+str(self.tol))
		dataMat = self.dataMat
		simDesign = self.simDesign
		histBins = self.histBins
		self.datComp = dataCompress.dataCompress(dataMat = dataMat,histBins=histBins,simDesign=simDesign,tol=self.tol)
		self.datComp.rowStd[self.datComp.rowStd==0] = 1e-3
		self.datComp.unitTrans()
		self.datComp.basisCompute()
		print('Data compression unit initialization finished')

	def InitializeGaussianProcess(self):
		"""
		Initialization function for constructing GPR unit.
		"""
		print('Initializing gaussian process unit.')
		kernel = 1*RBF(length_scale=np.ones(self.ndim), length_scale_bounds=(1e-2, 1e2))#+ WhiteKernel(noise_level=1e-2,noise_level_bounds=(1e-10,1e2))
		PCA_number = self.datComp.pca_weights.shape[0]
		print('Number of input simulation is '+str(self.simDesign.T.shape[0]))
		print('Number of PCA remain is '+str(PCA_number))
		self.gp = []
		for i in range(PCA_number): 
			self.gp.append(GaussianProcessRegressor(kernel = kernel,n_restarts_optimizer=20).fit(self.datComp.simDesign,self.datComp.pca_weights[i]))
		print('Gaussian process unit initialization finished')
	
	def GetPtheta(self,posterior_vector,hyperParam):
		"""
		Get normalized probability density function value at the given coordinate and hyperparameters.
		"""
		gp_distribution = self.GetDistribution(hyperParam).reshape(self.binShape)
		normalization = 1
		if self.histBins.ndim ==1:
			normalization *= np.gradient(np.unique(self.histBins))[0]
		else:
			for i in range(self.histBins.T.shape[0]):
				normalization *= np.gradient(np.unique(self.histBins.T[i]))[0]
		normalization *= np.sum(gp_distribution)
		p_theta = interpn(self.binAxes,gp_distribution,posterior_vector,bounds_error=False,fill_value=0)/normalization
		return p_theta

	def GetNtheta(self,posterior_vector,hyperParam):
		"""
		Get number of events value at the given coordinate and hyperparameters.
		"""
		gp_distribution = self.GetDistribution(hyperParam).reshape(self.binShape)
		n_theta = interpn(self.binAxes,gp_distribution,posterior_vector,bounds_error=False,fill_value=1e-20)
		return n_theta


	def GetNormalization(self,hyperParam,return_std=False):
		"""
		Get total rate of a given hyperparameter.
		"""
		if return_std==False:
			gp_distribution = self.GetDistribution(hyperParam)
			normalization = np.sum(gp_distribution)
			return normalization
		else :
			gp_distribution = self.GetDistribution(hyperParam,return_std=True)
			mean = np.sum(gp_distribution[0])
			upper = np.sum(gp_distribution[1])
			lower = np.sum(gp_distribution[2])
			return [mean,upper,lower]
		
 
	def GetDistribution(self,hyperParam,return_std=False):
		"""
		Get the distribution on the event parameter.
		"""
		N_PCA = self.datComp.pca_weights.shape[0]
		pca_predict = np.zeros(N_PCA)
		if return_std==False:
			for i in range(N_PCA):
			  pca_predict[i] = self.gp[i].predict([hyperParam],return_std=False)
			gp_distribution = np.power(10,self.datComp.rotate2full(pca_predict))
			return gp_distribution 
		else:
			upper_pca = np.zeros(N_PCA)
			lower_pca = np.zeros(N_PCA)
			for i in range(N_PCA):
				GPbuffer = self.gp[i].predict([hyperParam],return_std=True)
				pca_predict[i] = GPbuffer[0]
				upper_pca[i] = pca_predict[i]+GPbuffer[1]
				lower_pca[i] = pca_predict[i]-GPbuffer[1]	
			mean = np.power(10,self.datComp.rotate2full(pca_predict))
			upper = np.power(10,self.datComp.rotate2full(upper_pca))
			lower = np.power(10,self.datComp.rotate2full(lower_pca))
			return [mean,upper,lower]
	
	def GetBinEdge(self):
		axis = []
		shape = np.zeros(self.histBins.shape[0])
		for i in range(self.histBins.shape[0]):
			axis.append(np.unique(self.histBins[i])-np.gradient(np.unique(self.histBins[i]))[0])
			shape = len(axis[i])
			axis[i] = np.append(axis[i],np.gradient(axis[i])[0])
		return axis,shape
