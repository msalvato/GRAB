from setuptools import setup, find_packages
setup(
    name='grab_dataset',
    version='0.0.1',    
    description='GRAB Dataset',
    url='https://github.com/msalvato/GRAB',
    author='otaheri (packaging done by msalvato)',
    #author_email='shudson@anl.gov',
    license='Scientific Purposes',
    packages=find_packages(),
    install_requires=['numpy>=1.16.2',
                    'torch>=1.0.1.post2',
                    'torchgeometry>=0.1.2',
                    'smplx >=0.1.2',
                    'pillow',
                    'tqdm',
                    'pyrender>=0.1.23',
                    'trimesh>=2.37.6',
                    'PyYAML',                    
                      ],

    classifiers=[
        'Development Status :: Not for public',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',      
        'Programming Language :: Python :: 3.8',
    ],
)
