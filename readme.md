# UTP_Melon_Recognition

Project use to count the number of melon on a given video.  
This is used to count the number of melon in the first row situated in a greenhouse.

The old_detect.py file should not be used, less effecient, slower and cannot track melon between frames,
but is a good way to show how to scrapt frames with open-cv and draw yourself the bounding boxes.  

## Install
To use this code, you need first to install theses packages for the detect.py:

 - Ultralytics
 - Pytorch (with cuda to improve computation time by a lot)

And for the old_detect.py, you also need to install theses packages:

 - Pillow
 - Matlpotlib
 - Opencv-headless

## Model creation  

This model was created using the yolo cli tool in an annaconda environnement.
The dataset was made with the Roboflow online annotation tool. 187 images have been used to build the dataset.

### Train  

There are 3 dataset versions availables:  
 - V1 detect melons in the background and the front row
 - V2 detect melons only in the front row
 - V3 detect and sort the melons if they are in the front row or behind the front row

If you want to make your own model and use the dataset,  install the YOLO cli tool and run this command to make a basic model. Fine tunning can be achieved with other arguments.

``yolo train data=dataset/V2/data.yaml epochs=100 imgsz=640``  

## Launch
You can use the model with the following command, the model will use the latest model provided in this repo at ``/model/detect``:

``python detect.py "<path to your video>"``