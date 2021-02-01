MiTfAT
======

.. image:: https://img.shields.io/pypi/v/MiTfAT.svg
    :target: https://pypi.python.org/pypi/MiTfAT
    :alt: Latest PyPI version
.. image:: https://zenodo.org/badge/290846934.svg
   :target: https://zenodo.org/badge/latestdoi/290846934
.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0


Introduction
------------

`MiTfAT` is a scikit-learn-friendly Python library to analyse fMRI data, with a focus on molecular fMRI.

Manual can be found `in this pdf file <https://github.com/vahid-sb/MiTfAT/blob/master/docs/mitfat.pdf>`_.

There are already a few Python packages that are used by researchers to pre-process the fMRI time-series and then analyse them, for example [`fitlins`](https://fitlins.readthedocs.io/en/latest/), [`niworkflows`](https://github.com/nipreps/niworkflows), and [`NiBetaSeries`](https://joss.theoj.org/papers/10.21105/joss.01295) which focus on very specific points of the analysis workflow. Or even the more comprehensive library [`nilearn`](http://nilearn.github.io) that includes various visualization functionalities and machine learning tools to analyse fMRI data, but does not provide a ready-made framework to contain various information and measurements related to an experiment in molecular fMRI experiments. Hence the `MiTfAT` library was developed. It can be used for general fMRI time-series analysis, but in particular, signals obtained from molecular fMRI studies, i.e. the cases in which we measure the changes in concentration of molecules that might have been directly injected into the brain. The `MiTfAT` library incorporates all the information and data related to an experiment into a Python class object called `fmri_dataset`. And various attributes of this class can be used to identify all the data related to each experiment, and perform analyses on all. Such datasets can include various MRI measurements of the same subject, for example, T1-weighted and FISP signals measured almost simultaneously. Or the dataset can include many trials in which the same set of stimuli is presented or applied to a subject repeatedly.

The basic principle behind `MiTfAT` is that it imports all the relevant data of an fMRI experiment into an object/class of type `fmri_dataset`. The fMRI time-series are a member of this class and are stored as NumPy arrays. There are various functionalities available to analyse the data in a number of ways. They include:

- Clustering the time-series using K-means clustering. Clustering can be done based on values in all time-steps, or the mean value of each time-series or slope of a linear-regression passing through the time-series. And it can be applied to raw data or the normalized data. `MiTfAT` uses `scikit-learn` for all machine learning functionalities. Hence, K-means can be easily changed with any other clustering algorithm implemented in `scikit-learn`.

- Removing voxels with a low signal to noise ratio. This is done using a clustering algorithm in two stages. In the first stage, the algorithm removes the time-series corresponding to voxels in which signal to noise ratio is not high enough. And in the second stage, the time-series corresponding to the remaining voxels are clustered.

- Detrending. We can remove the general trends in time-series to make the transient changes more visible. As an example, if we cannot wait long enough until the concentration of our agent reaches the steady-state level, then transient variations in the signal caused by changes in experimental conditions, which is what we are interested in, might be obscured by such trends in the signal. `MiTfAT` can detrend the time series and give us a signal which looks like the one we might expect in a steady-state condition. And then, transient changes due to experimental design would be quantifiable.

- Interpolation of time-series under varying conditions. Assume say we have an experiment in which the total recording time is divided into 3 segments. In the second segment, we record under a different condition compared to the first and last segment. This can be, as an example, the occlusion of an artery which changes calcium concentration in the brain when the agent we have injected into the brain binds with calcium. Obviously, we want to quantify how the signal has changed in the second segment. In order to do so, we should interpolate the time-series in segment 2 based on values of segments 1 and 3, and then compare it with the actual measurements. `MiTfAT` provides such a functionality out of the box.

- Averaging over many trials. If we repeat an experiment many times, then it is usually of interest to average the measurements over all the trials. This can be useful if each measurement is noisy and we want to attenuate the effects of noise by averaging. `MiTfAT` provides such functionality.

Installation
------------
It is better to install `MiTfAT` in a new virtual environment. If you are using Anaconda Python, you can do the following:

.. code-block:: bash

   conda create -n env_mitfat
   conda activate env_mitfat


Then in your command prompt or bash, simply type:

.. code-block:: bash

   pip install mitfat


Or if you want to work with the latest beta-release, you can install directly from `this repository <https://github.com/vahid-sb/MiTfAT>`_.

Basics
------
The `MiTfAT` library incorporates all the relevant data and information about an experiment into a Python class of type `fmri_dataset`, and then the user can perform various analysis and visualisatin steps. In order to load the fMRI data, currently, the required information about data files and details of the experiment should be written down in a specified format in a config file, details of which will be discussed shortly. But if you want to get started with some sample data and know some of the features of the library, you can download two sample scripts and corresponding datasets from `here <https://github.com/vahid-sb/MiTfAT/blob/master/tests.zip>`_ . When you unzip the file, you can see a folder called ``tests`` in which you can see two python scrips. There are also two subfolder, each contains sample datasets that are used by each of the scripts. And you can also see three text files which are the config files used by scripts.

If you have installed `MiTfAT`, then you can run each of these scripts and the outputs they generate will be saved in new sub-folders inside the ``tests`` folder. Studying these two scripts can be quite informative and it is highly recommended for the users. If you want to use these samples scripts for your own data, you can simply edit the config files.

In the following chapters, main features of the code are explained. The figures you will see in the following chapter are generated using these two sample scripts.


Requirements
^^^^^^^^^^^^

 .. code-block:: bash

	"pandas",
	"numpy",
	"scipy",
	"matplotlib",
	"nibabel",
	"nilearn",
	"pathlib",
	"click",
	"seaborn",
	"openpyxl",


Compatibility
-------------

This code is tested under Python 3.7, and 3.8.

License
-------
GNU General Public License (Version 3).

Citation
--------
Please cite this code as follows:

Bokharaie VS (2019) "`MiTfAT`: A Python-based fMRI Analysis Tool", Zenodo. https://doi.org/10.5281/zenodo.3372365.

Citation
--------
Please use the CITATION.cff file.

This code was originally developed for a collaboration which led to the following publications:

Savić T. , Gambino G., Bokharaie V. S., Noori H. R., Logothetis N.K., Angelovski G., "Early detection and monitoring of cerebral ischemia using calcium-responsive MRI probes", PNAS, 2019.


Author
-------

`MiTfAT` is maintained by `Vahid Samadi Bokharaie <vahid.bokharaie@protonmail.com>`_.
