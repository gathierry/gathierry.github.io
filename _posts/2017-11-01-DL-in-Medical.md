---
layout: post
title: Summary of Deep Learning in Medical Imaging Application
tags: [computer vision, deep learning, medical image]
description: 
date: 2017-11-01
feature_image: images/2017-11-01-DL-in-Medical/title.jpg
---

This report is basically a summary for the paper _A Survey on Deep Learning in Medical Image Analysis_ [Litjens et al.](https://arxiv.org/abs/1702.05747) to introduce some deep learning techniques in medical image analysis. For more details, please refer to the original paper.

<!--more-->

## Introduction
The referenced paper is a survey on deep learning applied to medical image analysis including over 300 papers before February 1st, 2017 from PubMed, ArXiv and conferences such as MICCAI, SPIE, ISBI and EMBC. Papers that didn't report results on medical image data or only used standard feed-forward neural networks with handcrafted features are excluded. Figures below shows the distributions of model used, task addressed, imaging modality and anatomical application area in this survey.
<img src="/images/2017-11-01-DL-in-Medical/distribution.png" width="400px"/>

## Deep Learning Uses in Medical Imaging
### 1. Classification
#### 1.1 Image/exam classification
For this application, one or multiple images (an exam) are taken as input and a single diagnostic variable is output (e.g., disease present or not).

- **Multi-class grade assessment of knee osteoarthritis**: This is an application on Xray images. But could be used on surgery as well. It demonstrates the possibility and potential to assess osteoarthritis with deep learning method.
- **Alzheimer's disease based on Brain MRI**

#### 1.2 Object or lesion classification
Object classification focuses on the classification of a small (previous identified) part of the medical image.

- **Nodule classification in chest CT**: In fact, deep learning could be applied to any lesion classification problem, not only pulmonary nodule, but also liver lesion for instance.
- **Assess survival in patients suffering from high-grade gliomas**: This is an application on MRI.

### 2. Detection
#### 2.1 Organ, region and landmark localization
Anatomical object localization is an important preprocessing step in segmentation tasks or in the clinical workflow for therapy planning and intervention.

- **Landmark identification on distal femur surface**: This is an application on 2D MRI slices.
- **Anatomical regions localization**: This is usually on 3D CT volumes. Anatomy needed to localize could be heart, aortic arch and descending aorta, etc.
- **Carotid artery bifurcation detection in CT data**
- **Detection of the aortic valve in 3D transesophageal echocardiogram**
- **Detection up to 12 standardized scan planes in mid-pregnancy fetal ultrasound**
- **End-diastole and end-systole frames in cine-MRI of the heart**

As long as I am concerned, anatomy in images is usually of irregular shape so that applying an image segmentation could be more useful than just doing the detection.

#### 2.2 Object or lesion detection
The task consists of the localization and identification of small lesions in the full image space.

- **Nodules detection in x-ray images**: Applying convolutional neural network on this problem has been proposed in 1995. Nowadays, the data source is extended to other modalities.
- **Detecting micro-bleeds in brain MRI**
- **Detection of nodules in chest radiographs and lesions in mammography**: This kind of researches have a great marketing value since both pulmonary and breast cancer screening are very popular these days. A good algorithm can provide the doctor with advices and save time.

### 3. Segmentation
#### 3.1 Organ and substructure segmentation
The segmentation allows quantitative analysis of clinical parameters related to volume and shape such as cardiac and brain analysis.

- **Segment gray and white matter in a brain MRI data set**
- **Vertebral body likelihood maps**: This droves deformable models for vertebral body segmentation in MR images.
- **Segment brain MRI**
- **Segment the pectoral muscle in breast MRI**
- **Segment coronary arteries in cardiac CT angiography**: This sort of segmentation could help determine best phase to reconstruct in CT application.

#### 3.2 Lesion segmentation
Segmentation of lesions combines the challenges of object detection and organ segmentation.

- **Segment white matter lesions in brain MRI**

### 4. Registration
Registration of medical images is a common image analysis task in which a coordinate transform is calculated from one medical image to another. This could be a spatial alignment or a temporal one.

- **Assess the local similarity between CT and MRI images of the head**: Registration could be used in some cross-modality situations.
- **Asses the pose and location of an implanted object during surgery**: In surgery operations, registration could be useful since the situation is changing but the imaging system must keep stable.
- **Prior/current registration in brain MRI**

### 5. Other tasks
#### 5.1 Image generation and enhancement
Image generation and enhancement methods could be ranging from removing obstructing elements in images, normalizing images, improving images quality, data completion an pattern discovery.

- **regular and bone-suppressed Xray**
- **3T and 7T brain MRI**: This could also be low-dose and high-dose CT or any other applications in various modalities.
- **CT from MRI**
- **Reconstruct high-resolution cardiac MRI from one or more low-resolution input MRI volumes**
- **Inferring advanced MRI diffusion parameters from limited data**
- **Denoising in DCE-MRI time-series**

The image generation task is usually to "make up" an image. This can be widely used for natural image but takes a great risk for medical image. I think that it's not very likely to see such a feature in medical imaging products in the near future.

## Conclusion
Deep learning is widely used in researches on medical imaging. In this report, we checked the most popular deep learning uses in medical imaging. Some of them are still prototypes in the lab whereas some of them are already integrated into products. Once we desire to solve a problem with deep learning technique, methods in this summary can inspire us and help to assess the feasibility.