# Nuclei-Segmentation-in-Histopathology-Images-Internship-Work

**This repository includes the brief report of the work done during the internship at MeDAL IIT Bombay.**
<br/>


The task was to review, modify and perform various nuclei segmentation methods by taking inspiration from different segmentation tasks being performed globally and achieve a good AJI score over MoNuseg dataset.

### About MonuSeg dataset:-

The MoNuSeg (Multi-organ Nuclei Segmentation) dataset was obtained by carefully annotating tissue images of several patients with tumors of different organs and who were diagnosed at multiple hospitals. This dataset was created by downloading H&E stained tissue images captured at 40x magnification from TCGA archive. The complete annotation and preprocessing of these H&E images has been done at MeDAL IIT Bombay only.

### Experiment 1:-

We first analysed the method adopted in the paper [Deep advesarial training for multi-organ Nuclei segmentation in Histopathology images](https://arxiv.org/abs/1810.00236).
This paper is heavily based on the GANs for the nuclei segmentation purpose. First of all, to avoid the problem of small size of the data, we first tried to create H&E image-mask pair using Cycle-GAN and randomly generated masks.
Using these artificially generated image-mask pairs, nuclei segmentation is performed using conditional GANs (pix2pix image translation).
The AJI as claimed by the authors over MoNuSeg dataset is 0.721.

### Our Conclusions:-
1) The complete training of the model over MoNuSeg dataset has been performed. The post processing of the results is underway.      As soon as it is done, we can report the AJI obtained for this independent training.
2) As the lot contours in output images are incomplete, background is noisy it needs heavy post processing.
3) According to some other recent papers, conditional GANs (which gives us image to image translation from simple to complex images) can’t be used for the segmentation tasks as segmentation is complex to simple image mapping. The same thing we have discovered here, while implementing this paper.
4) Hence, we moved to a simple end to end supervised network, whose loss function focuses more on the small region occupied by the boundaries.


### Experiment 2:- 

The next paper which we followed is [CE-Net: Context Encoder Network for 2D Medical Image Segmentation](https://arxiv.org/abs/1903.02740).

This paper proposes similar network architecture as that of U-Net adding DAC and RMP block to it. The main function of these extra blocks is to extract features from shapes of every possible size, so that this network architecture can be generalised for segmentation of different types of medical images such as optic disc segmentation, lung segmentation, cell boundary segmentation etc. The loss used here in Dice coefficient loss. 

**We tried this model on MoNuSeg dataset.** 
1. First we tried it to train for localisation and detection task. The results are quite well. The result images will be made available soon.
2. Then we tried it to train for nuclei segmentation directly. We found that it is having problem in distinguishing between overlapping nuclei. 
3. Hence we have come up with the way to use task 1 training for the benefit of the task 2. The work on this new architecture is in progress. Results will be made available soon. 


### Experiment 3:-

The results of the paper [Mask-RCNN](https://arxiv.org/abs/1703.06870) are quite remarkable for the task of instance segmentation. Our work would have been incomplete without exploring this model for the Nuclei segmentation task. 

We trained the Mask-RCNN model from scratch and got the results. The output images were noiseless with fine boundaries. We were able to calculate it's AJI straight away, which came out to be 0.5598

Still we will give attempts to train this model by tuning it's hyperparameters, and continuosly update the results.



**Note:-**  Images are yet to be included.
