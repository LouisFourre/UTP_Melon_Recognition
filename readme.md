# UTP_Melon_Recognition

Project use to count the number of melon on a given video.  
This is used to count the number of melon in the first row situated in a greenhouse 

## Install
To use this code, you need first to install theses packages for the detect.py:

 - Ultralytics
 - Pytorch (with cuda to improve computation time by a lot)

And dor the old_detect.py, you also need to install theses packages:

 - Pillow
 - Matlpotlib
 - Opencv-headless

## Model creation  

This model was created using the yolo cli tool in an annaconda environnement.
The dataset was made with the Roboflow online annotation tool. 187 images have been used to build the dataset.

### Train  

``yolo train data=dataset/v1 epochs=100 imgsz=640``  

## Launch
You can lauch this model with this commande

``python detect.py "<path to your video>"``