---
title: 'MiTfAT: A Python-based Analysis Tool for Molecular fMRI, Made in Tuebingen.'
tags:
  - Python
  - Cpmputational Neuroscience
  - fMRI Analysis
  - Machine Learning
  - Time-Series Analysis
authors:
  - name: Vahid S. Bokharaie^[Corresponding Author.]
    orcid: 0000-0001-6544-7960
    affiliation: 1
affiliations:
 - name: Max Planck Institute for Biological Cybernetics, Tuebingen, Germany
   index: 1
date: 20 October 2020
bibliography: paper.bib
---


# Summary
 
Functional Magnetic Resonance Imaging, fMRI, is a technique used in neuroscience to measure brain activity based on any signal that can be measured in an MRI scanner. Normally, fMRI is used to detect changes associated with blood flow. But it can also be used to detect changes in concentrations of molecules with different magnetic properties which are directly injected into the brain of a subject. 

Regardless of the signal that is measured in fMRI recording, from a computational point of view, fMRI recordings will result in a number of time-series. And then those time-series should be analysed to find the answers to various questions of interest. The length of the time-series depends on the number of time-steps in which we have measured the signals, and their number depends on how-many voxels we have measured (a voxel is a 3-dimensional pixel or the unit of volume in which each fMRI signal is measured). What is the size of each of these voxels depends on the magnetic flux density of the MRI scanner, measures in Tesla (T). The higher is the magnetic flux density, the smaller are the voxels and the higher is the spatial resolution of the measurement. Hence, we end up with one time-series for each of the voxels arranged in a 3-dimensional structure. One characteristic of the fMRI measurements is that while they can have a high spatial resolution, and while they allow us to measure brain activity not only in the cortices, but also in deeper regions of the brain, but their temporal resolution is not normally high. And that can provide challenges for researchers who need to analyse the fMRI data.

# Statement of need

`MiTfAT` is a scikit-learn-friendly Python library to analyse fMRI data, with a focus on molecular fMRI experiments. It was primarily developed for the study which is presented in [@savic:2019].
   
   There are already a few Python packages that are used by researchers to pre-process the fMRI time-series and then analyse them, for example [@esteban:2019], and [@kent:2019] which focus on very specific points of the analysis workflow. Or even a more comprehensive library such NiLearn [@abraham2014machine] that includes various visualization functionalities and machine learning tools to analyse fMRI data, but does not provide a ready-made framework to contain various information and measurements related to an experiment in molecular fMRI experiments. Hence the `MiTfAT` library was developed. It can be used for general fMRI time-series analysis, but in particular, signals obtained from molecular fMRI studies, i.e. the cases in which we measure the changes in concentration of molecules that might have been directly injected into the brain. The `MiTfAT` library incorporates all the information and data related to an experiment into a Python class object called `fmri_dataset`. And various attributes of this class can be used to identify all the data related to each experiment, and perform analyses on all. Such datasets can include various MRI measurements of the same subject, for example, T1-weighted and FISP signals measured almost simultaneously. Or the dataset can include many trials in which the same set of stimuli is presented or applied to a subject repeatedly. 
   
   The basic principle behind `MiTfAT` is that it imports all the relevant data of an fMRI experiment into an object/class of type `fmri_dataset`. The fMRI time-series are a member of this class and are stored as NumPy arrays. There are various functionalities available to analyse the data in a number of ways. They include:
   
   - Clustering the time-series using K-means clustering. Clustering can be done based on values in all time-steps, or the mean value of each time-series or slope of a linear-regression passing through the time-series. And it can be applied to raw data or the normalized data. `MiTfAT` uses `scikit-learn` for all machine learning functionalities. Hence, K-means can be easily changed with any other clustering algorithm implemented in `scikit-learn`. 
   
   - Removing voxels with a low signal to noise ratio. This is done using a clustering algorithm in two stages. In the first stage, the algorithm removes the time-series corresponding to voxels in which signal to noise ratio is not high enough. And in the second stage, the time-series corresponding to the remaining voxels are clustered to identify the distribution pattern of the contrast agent. 
   
   - Detrending. We can remove the general trends in time-series to make the transient changes more visible. As an example, if we cannot wait long enough until the concentration of our agent reaches the steady-state level, then transient variations in the signal caused by changes in experimental conditions, which is what we are interested in, might be obscured by such trends in the signal. `MiTfAT` can detrend the time series and give us a signal which looks like the one we might expect in a steady-state condition. And then, transient changes due to experimental design would be quantifiable.
   
   - Interpolation of time-series under varying conditions. Assume say we have an experiment in which the total recording time is divided into 3 segments. In the second segment, we record under a different condition compared to the first and last segment. This can be, as an example, the occlusion of an artery which changes calcium concentration in the brain when the agent we have injected into the brain binds with calcium. Obviously, we want to quantify how the signal has changed in the second segment. In order to do so, we should interpolate the time-series in segment 2 based on values of segments 1 and 3, and then compare it with the actual measurements. `MiTfAT` provides such a functionality out of the box. 
   
   - Averaging over many trials. If we repeat an experiment many times, then it is usually of interest to average the measurements over all the trials. This can be useful if each measurement is noisy and we want to attenuate the effects of noise by averaging. `MiTfAT` provides such functionality. 

# Examples
`MiTfAT` repository includes a manual that contains many examples of the various capabilities of the library. It can be found [here](https://github.com/vahid-sb/MiTfAT/tree/master/docs/mitfat.pdf)

There are also two scripts in `tests` folder of the repository ([here](https://github.com/vahid-sb/MiTfAT/tree/master/tests/)), accompanied with sample datasets, which you can run to see sample outputs of the library. 
 
## References

