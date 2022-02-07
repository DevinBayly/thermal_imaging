# thermal_imaging

This repository contains the work and information related to the Thermal Imaging project requested by Ed Wellman and Brad Ross from the University of Arizona. The goal of my involvement in the project was to write code to do rudimentary computer vision on the large collection of videos that had been taken from different open mines in North America. The code written is intended to run in HPC environments, and this document will walk the reader through the process of initiating a processing session. This document will also serve as a navigation aide for finding previous research done to support the computer vision aspects of this project. 

## Resources

- Issue tracker with research notes:
- HPC documentation: The base documentation is https://public.confluence.arizona.edu/display/UAHPC. The hpc consultants offer drop in hours for folks with questions on wed 2-4 or via support ticket https://public.confluence.arizona.edu/display/UAHPC/Getting+Help. This project is leveraging slurm array job submission which is described here https://public.confluence.arizona.edu/display/UAHPC/Running+Jobs+with+SLURM and https://public.confluence.arizona.edu/display/UAHPC/Running+Jobs+with+SLURM#RunningJobswithSLURM-ArraySubmission.
- Singularity Containers: The dependencies of the code are satisfied by a singularity container inside which all the code runs. These are great tools for research on and off supercomputers and more can be read about them here https://sylabs.io/singularity. Our university hosts a training for researchers interested in learning more about singularity containers at https://cyverse.org/cc. 
- OpenCV python: Open CV is a computer vision package that makes it easy to do many tasks without having to implement your own computer vision algorithms https://docs.opencv.org/4.x/. Strongly suggest that people read about background segmentation https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html and blob detection https://learnopencv.com/blob-detection-using-opencv-python-c/. 
- Data Visualization Log Plotter: This is a tool that is meant to assist in viewing the outputs of a video processing job. https://devinbayly.github.io/thermal_imaging/. There will be instructions provided on how to use this in this document.
- Zoom recording links: Many of the meetings we had over the course of the last 6 months are recorded here https://github.com/DevinBayly/thermal_imaging/issues/26.

## Process Outline 

![](./Drawing1.svg)

## Setup Steps Walkthrough

### Upload Videos to HPC

The steps for this are covered in the latest Zoom recording. What you need to do is sign into https://www.globus.org/ and use the https://app.globus.org/file-manager file manager to initiate transfer between the machine that has the external hard drive plugged in and the UA HPC FileSystems. Ensure that globus is running on the desktop machine before using the file browser. 

### Clone the Github Repo

All the code necessary to process the videos is kept in version control on github. The repo that hosts this readme also comes with scripts for starting super computer jobs and python scripts to person computer vision tasks on mine videos. In order to get the code you must clone the repository in an interactive session on the HPC. 

#### Using OOD for starting an interactive session

Navigate to ood.hpc.arizona.edu/ and fill in the web auth details. You will then see 
![](https://user-images.githubusercontent.com/11687631/152853352-ca8fe49e-0f3b-4530-ad71-f6c3e0eca829.png)
click on the `clusters` tab at the top and select shell. This brings you to a login node on the HPC using the browser for your shell session. Then we will submit the lines of code shown in the following image:

*Note there will be several differences in what you type. You will not be using `/xdisk/chrisreidy/baylyd`, instead something like `/xdisk/bjr/`. Second your command would be `interactive -a bjr` not `interactive -a visteam` because bjr is the allocation for this project on the HPC. Also you won't encounter the error shown here because there won't be an existing directory called `thermal_imaging`.*

![](https://user-images.githubusercontent.com/11687631/152854276-b3f059e7-c088-415d-a7cd-5462a7144fdf.png)


Here are the individual commands in a list to follow
```
elgato
interactive -a bjr
cd /xdisk/bjr
git clone https://github.com/DevinBayly/thermal_imaging.git
```

You should now have a folder called `thermal_imaging` at this absolute path `/xdisk/bjr/thermal_imaging`. This will be where you complete the next step as well.



### Retrieve the singularity container

The singularity container is  currently hosted by github at https://github.com/DevinBayly?tab=packages, but you must use singularity to pull it down to the HPC. This is a step that you will only have to do once to get a copy of the container on the HPC. First ensure you are in `/xdisk/bjr/thermal_imaging/` then run these commands.

![](https://user-images.githubusercontent.com/11687631/152855249-cf2668c2-d456-4b2b-889e-28dc8557baa7.png)

```
singularity pull oras://ghcr.io/devinbayly/thermal.sif:latest
mv thermal.sif_latest.sif thermal_imaging.sif
```

This will make sure you have the singularity container in the correct folder for being used in the batch processing to come. 
