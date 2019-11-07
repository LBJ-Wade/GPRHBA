from setuptools import setup
from setuptools import find_packages

def readme():
	with open('README.rst') as f:
		return f.read()

setup(name='GPRHBA',
	version='0.1',
	description='HBA with GPR and PCA',
	long_description=readme(),
	classifiers=[
		'Development Status :: 3 - Alpha',
		'License :: OSI Approved :: MIT License',
		'Programming Language :: Python :: 2.7',
		'Topic :: Gravitational Wave :: Physics',
	],
	keywords='Gravitational wave',
#	url='http://github.com/storborg/funniest',
	author='Kaze W. K. Wong',
	author_email='kazewong@jhu.edu',
	license='MIT',
	packages=find_packages(exclude=['Others_code','notebook','scripts']),
	install_requires=[
		'numpy',
		'scipy',
		'sklearn',
	],
	include_package_data=True,
	zip_safe=False)
