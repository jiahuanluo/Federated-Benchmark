<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
## FedVision
**FedVison dataset** is created jointly by WeBank and ExtremeVision to facilitate the advancement of academic research 
and industrial applications of federated learning.

### The FedVision project

* Provides images data sets with standardized annotation for federated object detection.
* Provides key statistics  and systems metrics of the data sets.
* Provides a set of implementations of baseline for further research.

### Datasets
We introduce two realistic federated datasets.
 
* **Federated Street**, a real-world object detection dataset that annotates images captured by a set of street cameras 
based on object present in them, including 7 classes. In this dataset, each or every few cameras serve as a device.

* **Federated River**, a dataset built from images of a set of river cameras. Here, every camera focuses on part of 
a river and each or every few of them serve as a device. The target of this dataset is to locate the garbage on the river. 

 | Dataset | Number of devices | Total samples | Number of class| 
 |:---:|:---:|:---:|:---:|
 | Federated Street | 5, 20 | 956 | 7 |
 | Federated River | 16 | 11072 | 1 |
 
### File descriptions

* **federated_street.tar.gz** contains the image data and ground truth for the train and test set of the street data set.
    * **Images**: The directory which contains the train and test image data.
    * **train_label.json**: The annotations file is saved in json format. **train_label.json** is a `list`, which 
    contains the annotation information of the Images set. The length of `list` is the same as the number of image and each value
    in the `list` represents one image_info. Each `image_info` is in format of `dictionary` with keys and values. The keys 
    of `image_info` are `image_id`, `device1_id`, `device2_id` and `items`. We split the street data set in two ways. For the first, we
    split the data into 5 parts according to the geographic information. Besides, we turn 5 into 20. Therefore we have `device1_id` and
     `device2_id`. It means that we have 5 or 20 devices. `items` is a list, which may contain multiple objects.  
    [  
     &emsp;    {  
     &emsp;&emsp;    `"image_id"`: the id of the train image, for example 009579.  
     &emsp;&emsp;    `"device1_id"`: the id of device1 ,specifies which device the image is on.   
     &emsp;&emsp;    `"device2_id"`: the id of device2.    
     &emsp;&emsp;    `"items"`: [  
     &emsp;&emsp;&emsp;       {  
     &emsp;&emsp;&emsp;&emsp;          `"class"`: the class of one object,  
     &emsp;&emsp;&emsp;&emsp;          `"bbox"`: ["xmin", "ymin", "xmax", "ymax"], the coordinates of a bounding box  
     &emsp;&emsp;&emsp;       },  
     &emsp;&emsp;&emsp;       ...  
     &emsp;&emsp;&emsp;       ]  
     &emsp;     },  
     &emsp;     ...  
    ]
    * **test_label.json**: The annotations of test data are almost the same as of the **train_label.json**. The only difference between them is that 
    the `image_info` of test data does not have the key `device_id`.  
   

* **federated_river.tar.gz** contains the image data and ground truth for the train and test set of the river data set.

    * **Images**: The directory which contains the train and test image data.
    * **train_label.json**: The annotations file is saved in json format. **train_label.json** is a `list`, which 
    contains the annotation information of the Images set. The length of `list` is the same as the number of image and each value
    in the `list` represents one image_info. Each `image_info` is in format of `dictionary` with keys and values. The keys 
    of `image_info` are `image_id`, `device_id` and `items`. `items` is a list, which may contain multiple objects.  
    [  
     &emsp;    {  
     &emsp;&emsp;    `"image_id"`: the id of the train image, for example 009579.  
     &emsp;&emsp;    `"device_id"`: the id of device ,specifies which device the image is on.   
     &emsp;&emsp;    `"items"`: [  
     &emsp;&emsp;&emsp;       {  
     &emsp;&emsp;&emsp;&emsp;          `"class"`: the class of one object,  
     &emsp;&emsp;&emsp;&emsp;          `"bbox"`: ["xmin", "ymin", "xmax", "ymax"], the coordinates of a bounding box  
     &emsp;&emsp;&emsp;       },  
     &emsp;&emsp;&emsp;       ...  
     &emsp;&emsp;&emsp;       ]  
     &emsp;     },  
     &emsp;     ...  
    ]
    * **test_label.json**: The annotations of test data are almost the same as of the **train_label.json**. The only difference between them is that 
    the `image_info` of test data does not have the key `device_id`.  
   
### Evaluation
We use he standard <u>[PASCAL VOC 2010](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/devkit_doc_08-May-2010.pdf)</u> mead Average Precision (mAP) to evaluate (mean is taken over per-class APs).  
To be considered a correct detection. the overlap ratio $$ a $$