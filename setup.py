from distutils.core import setup

setup(
	name='RobustSTL',
	version='0.1.1',
	author='Doyup Lee',
	
	packages=['robuststl'],
	license='MIT License',
	description='Robust Seasonal Decomposition',
	long_description=open('README.md').read(),
	install_requires=[
		'cxvopt >= 1.2.3',
		'matplotlib >= 2.2.2',
		'numpy >= 1.14.2',
		'pathos >= 0.2.3',
	],
)
