---
layout: post
title: Challenge - Sleep Stage Classification
tags: [machine learning, signal processing]
description: How to predict the sleep stage using electro-encephalogram (EEG) and accelerometer?
date: 2016-06-29
feature_image:
---
(challenge site : https://challengedata.ens.fr/en/challenge/1/sleep\_stages\_classification.html)

## Challenge context

According to the American Agency of Sleep Medicine, there exists 5 sleep stages, called Wake, N1, N2, N3 and REM.

Dreem headband is able to aquire EEG signals using frontal electrods as well as accelerometer data. Looking at these values, a doctor can determine the different sleep stages along a night. Nevertheless, this task is quite fastidious and time consuming.

We had our own doctor label nights, and aim now to automatically classify windows of 15 seconds.

## Challenge goals

Brain in general and sleep in particular involve complex processes.
The goal of this challenge is to be able to predict accurately sleep stages according to 15 seconds of EEG and accelerometer signals.

## Data description

### Input

The training dataset is composed of 31129 windows of 15 seconds of nights. The first row will give you the headers, and the first column corresponds to IDs of the windows. Each points is in dimension 4203, distributed this way :

- SLEEPER_ID (int) : id of the sleeper
- SIGNAL_ID (int) : id of the signal. You have full nights at your disposal, and are able to rebuild it.
_ IDX_IN_NIGHT (int) : index of the window in the night. Please note that each night may not be continuously present as we withdrew bad quality signals
- 3750 * EEG (float) : 15-second electro-encephalogram signal, sampled at 250Hz. This is the raw signal aquired by electrods, where we only cut 50 and 62,5Hz frequencies and applied a high-pass filter at 0,4Hz
- 150 * ACC_X (float) : 15-second of accelerometer data in direction x. We initially had 3750 points (250Hz), but convolved with a gaussian and then subsampled at 10Hz
- 150 * ACC_Y (float) : idem in direction y
- 150 * ACC_Z (float) : idem in direction z

### Output

The training labels correspond to the 5 acknowledged sleep stages (all integers) :

- 0 : Wake
- 1 : N1, sometimes said "light sleep" or "somnolence". Quite rare along the night
- 2 : N2, sometimes said "intermediate sleep", the major stage
- 3 : N3, or "deep sleep"
- 4 : REM, for "Rapid eye movement", also said "paradoxical sleep". This is the stage when dreams occure

## Our approach

### Pre-processing

Given signal time series as input, we should first of all pre-process the signal and extract features. 

For EEG, we have 15-second signal sampled at 250Hz. And we desire to split the signal into Delta (below 3.5Hz), Theta (4-7Hz), Alpha (8-13Hz) and Beta (14-30Hz) bands. Particularly, Alpha, Theta and Delta correspond to stage 1, 2 and 3.

Fourier transformation is used to the serie so that we can obtient
the spectral density for 0-250Hz. The energy, max/min/mean/std value of each band, the ratio of energy in the Alpha band and the combined power in Delta and Theta bands, the ratio of energy in the Delta band and the combined power in Alpha and Theta bands, the ratio of energy in Theta band and the combined power in Delta and Alpha bands are calculated and regarded as features. In fact, many articles used wavelets which is able to keep the information both of time and of frequency. But here, we use Fourier transformation since it is easier to implement.

For ACC, we also take their mean/std/max/min value as features. In this case, we have a classfication dataset with nearly 30 features.

### Machine learning

SVM and random forest are used to solve this question. As far as we are concerned, SVM with linear kernel is quite efficient. But it's definitely possible that there are other methods being able to achieve better scores.


## Bibliography

[1] Mohammad Mikaeili, Edson Estrada, Homer Nazeran, Farideh Ebrahimi, "Automatic sleep stage classification based on EEG signals by usig neural networks and wevelet packet coefficients," in annual international IEEE EMBS conference, Vancouver, 2008.

[2] Rosalind Picard Akane Sano, "Comparaison of sleep-wake classification using electroencephalogram and wrist-worn multimodal sensor data," , 2004.




