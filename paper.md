---
title: 'MiTfAT: A Python-based scikit-learn-friendly fMRI Analysis Tool, Made in Tuebingen.'

tags:
  - Python
  - fMRI Analysis
  - Machine Learning
  - Time-Series Analysis

authors:
  - name: Vahid S. Bokharaie
    orcid: 0000-0001-6544-7960
    affiliation: "1" # (Multiple affiliations must be quoted)

affiliations:
 - name: Max Planck Institute for Biological Cybernetics, Tuebingen, Germany
   index: 1

date: 20 October 2020

bibliography: paper.bib

# Summary
 
Functional Magnetic Resonance Imaging, fMRI, is a technique used in Neuroscience to measure brain activity based on any signal that can be measured in an MRI scanner. Normally, fMRI is used to detect changes associated with blood flow. But it can also be used to detect changes in concentrations of molecules with different magnetic properties which are directly injected into the brain of a subject. 

Regardless of the signal that is measured in fMRI recording, from a computational point of view, fMRI recordings will result in a number of time-series. And then those time-series should be analysed to find the answers to various questions of interest. The length of time-series depends on the number of time-steps in which we have measured the signals. And the number of these time-series depends on how-many voxels we have measured. A voxel is a 3-dimensional pixel or the unit of volume in which each fMRI signal is measured. What is the size of each of these voxels depends on the magnetic flux density of the MRI scanner, measures in Tesla (T). The higher is the magnetic flux density, the smaller are the voxels and the higher is the spatial resolution of the measurement. Hence, we end up with one time-series for each of the voxels arranged in a 3-dimensional structure. One characteristic of the fMRI measurements is that while they can have a high spatial resolution, and while they allow us to measure brain activity not only in the cortices, but also in deeper regions of the brain, but their temporal resolution is not normally high. Hence, we need the algorithms we use to analyse them might be very different than the ones we use to analyse time-series with higher temporal resolutions. And these algorithms can get complicated if we go beyond the very simple statistical analyses of the fMRI data.

# Statement of need

`MiTfAT` is a scikit-learn-friendly Python library to analyse fMRI data. It was primarily developed for the study which is presented in [[1]](#1).

There are a few software packages that are commonly used by researchers to pre-process the fMRI time-series and then analyse them. But the decision to develop a new library which eventually led to `MiTfAT` library is motivated by a few reasons. The main reason is that all the commonly used fMRI analysis packages come with so many belts and whistles that it usually takes the user a long time to find out which parts of each software package is what he/she need. Another reason was that none of the commonly used software packages I could find was written in Python, which is my programming language of choice. And lastly, the analysis methods I needed for the problems I had in mind were not available in any of those software packages. 

Hence the `MiTfAT` was developed. It is designed to be used for general fMRI time-series analysis, but in particular, signals obtained from molecular fMRI studies, i.e. the cases in which we measure the changes in concentration of molecules which might have been directly injected into the brain (which happens when the molecule is too big to pass through blood-brain-barrier).

The basic principle behind `MiTfAT` is that it imports all the relevant data of an fMRI experiment into a class object. The fMRI time-series are a member of this class and are stores as NumPy arrays. There are various functionalities available to analyse the data in a number of ways. They include:

- Clustering the time-series using K-means clustering. Clustering can be done based on values in all time-steps, or the mean value of each time-series or slope of a linear-regression passing through the time-series. And it can be applied to raw data or the normalised data. `MiTfAT` uses `scikit-learn` for all machine learning functions. Hence, K-means can be easily changed with any other clustering algorithm implemented in `scikit-learn`. 

- Hierarchical clustering; which applies clustering in two stages. In the first step, the algorithm removes the time-series corresponding to voxels in which signal to noise ratio is not high enough. And in the second step, the time-series corresponding to the remaining voxels are clustered. 

- Detrending. We can remove the general trends in time-series to make the transient changes more visible. As an example, if we cannot wait long enough until the concentration of our agent reaches steady-state level then transient variations in the signal caused by changes in experimental conditions, which is what we are interested in, might be obscured by such trends in the signal. `MiTfAT` can detrend the time series and give us a signal which looks like the one we might expect in steady-state. And then, transient changes due to experimental design would be quantifiable.

- Interpolation of time-series under varying conditions. Let's say we have an experiment in which the total recording time is divided into 3 segments. In the second segment, we record under a different condition compared to the first and last segment. This can be, as an example, the occlusion of an artery which changes calcium concertation in the brain when the agent we have injected into the brain binds with calcium. Obviously, we want to quantify how the signal has changed in the second segment. In order to do so, we should interpolate the time-series in segment 2 based on values of segments 1 and 3, and then compare it with the actual measurements. `MiTfAT` provides such a feature for us. 

- Averaging over many trials. If we repeat an experiment many times, then it is usually of interest to average the measurements over all the trials. This can be useful if each measurement is noisy and we want to attenuate the effects of noise. `MiTfAT` allows us to do so. 

# Examples
`MiTfAT` repository includes a manual which contains many examples of the various capabilities of the library. It can be found [here] (
https://github.com/vahid-sb/MiTfAT/blob/master/docs/mitfat.pdf)
 
## References
<a id="1">[1]</a> 
Savić T. , Gambino G., Bokharaie V. S., Noori H. R., Logothetis N.K., Angelovski G., (2019). 
Early detection and monitoring of cerebral ischemia using calcium-responsive MRI probes. 
Proceedings of the National Academy of Science of the USA (PNAS).

