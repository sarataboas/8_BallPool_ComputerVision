# 8_BallPool_ComputerVision


## **Task 1**

Complete later

## **Data**
Dataset source : https://universe.roboflow.com/bachelorthesis/8-ball-pool-l530o

The dataset is already splitted in Train, Validation and Test. 

Total available images: 247
- Train split: 207
- Validation split: 20 
- Test split: 20



#### **About 8-Ball Pool Dataset**
- Pool dataset collected for a Bachelor Thesis regarding object detection on pool balls. All images are screenshots from various YouTube videos of professional pool championships.

- Images are labeled starting from 1. Images with labels like "1a", "1t", "3f" were meant for a different experiment, where we studied the accuracy of our homography transformation from "some" view to top-view. Each subscript "t", "f" and "a" means:
    - "t": top-view
    - "f": front-view
    - "a": diagonal-view

**Data Version**: Downloaded the YOLOv11 dataset version (contains the images and the corresponding annotations)

**Important**: in the downloaded data, all images are inside the train/ folder -> necessary to separate them using the repository splits

#### **Extra Datasets (future work)**
- https://universe.roboflow.com/nidacorian-protonmail-com/pool-billiard
- https://universe.roboflow.com/mark-dj0yk/pool-balls-detection-srlqi
- https://universe.roboflow.com/pool-ball-detection/pool-ball-detection-6lfd9

## **Task 2**

### **Count the total number of balls in an image** [cnn_pipeline](models/cnn_pipeline.py)

- Input: list of images in a JSON file. The input file follows the structure below:
```json
{
    "image_path": [
        "development_set/106_png.rf.28ee53acf89d9e7f17b2fb26185597a0.jpg",
        ...,
    ]
}
```

- Output: the pipeline should output a list of results in a JSON file

- Model: CNN architecture

    - Extra: quantitative comparison (with adequate metrics) different architectures


## **Task 3**

### **Ball Detection**
- At least one model for ball classification 
    - Extra: quantitative comparison (with adequate metrics) of different architectures

### **Table Retrieval**

- Select the most similar pooling table from the training data
    - Evaluation: Qualitative evaluation with some (good and bad) results are enough




