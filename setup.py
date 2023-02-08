from setuptools import setup

setup(
    name='deepDNAshape',
    version='0.1.0',    
    description='Deep DNAshape',
    url='',
    author='Jinsen Li  ',
    author_email='jinsenli@usc.edu',
    license='BSD 2-clause',
    packages=['deepDNAshape'],
    install_requires=['tensorflow>2.0',
                      'numpy',                     
                      ],
    scripts=['bin/deepDNAshape'],
    include_package_data=True,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
