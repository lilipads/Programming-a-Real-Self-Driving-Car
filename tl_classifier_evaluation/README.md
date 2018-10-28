#### Background

Since testing the car in the simulator is expensive and long stretches of the road do not have traffic lights to detect, here we created a close to exhaustive repository for all traffic images in the simulator. By evaluating the classifier on this dataset, we will have a better understanding of the performance of the traffic light classifier in a time-efficient manner.

In addition to simulator data, there are also real world images. These are frames extracted from the "Traffic Light Detection Test Video" download (traffic_light_bag_files.zip) provided by Udacity. More specifically, the data is from the just_traffic_light.bag file.

#### Data generation

The car drove 4 loops in the simulator, saving camera images when it was close to a traffic light (< 200 waypoint indices). The images are named in the format of: <traffic_light_waypoint_index>-<car_waypoint_index>-<traffic_light_state>.jpg. So for example, when the car is at waypoint 123, the traffic light is at waypoint 753, and it is red (0), the file name is 753-123-0.jpg. The images were generated with a modified version of tl_detector.py that essentially saved images instead of performing inference.

Most of the traffic lights have images of all its three states (green, yellow, red) at various distances from the car. There are 2520 images in total.

Real world images were extracted using a similar approach.

#### How to evaluate the classifier:

1. unzip \<one of the zip files\>
2. cd ../ros/src/tl_detector/
3. vi light_classification/tl_classifier.py (set desired model_path manually)
4. vi tl_evaluation.py (set desired DATA_DIR)
5. python tl_evaluation.py
6. cat ../../../wrongly_classified_image.csv
