from setuptools import setup, find_packages

setup(name="business-individual-classifier", packages=find_packages(), install_requires=['pytest','sklearn','PyYAML', 'faker', 'confuse', 'keras',
                                                                                         'numpy', 'pandas', 'h5py', 'tensorflow==2.6.4'])
