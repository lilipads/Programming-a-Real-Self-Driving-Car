### Udacity Self-Driving Car Engineer nanodegree - Capstone Project
#### "Programming a Real Self-Driving Car"

Team Members: David Svanlund, Lili Jiang, Yunjae Choi, Li Gen, Sandeep Paulraj

### Background
In this capstone project, we implemented various aspects of a self-driving car using the [Robot Operating System (ROS)](http://www.ros.org) and [Autoware](https://autoware.ai). The code both powers a virtual car to drive on a simulated road in a simulator environment and also a real Udacity self-driving car to drive in a parking lot. The car will be able to follow a pre-defined trajectory with controlled acceleration / speed, and also will be able to respond to traffic lights. 

See demo of a self driving car loaded with our code!

![demo](imgs/demo.gif)

The key aspects of the implementations are: a) traffic light detection, b) waypoint update, c) controller. We discuss our implementation in detail in sections below.

### Traffic Light Detection

This module is tasked with accurately determining the color of each traffic light the car comes across. This is the part we spent most of our time working on.


#### Deep learning model
We used the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) along with pre-trained models from the [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) as the starting point, and fine tuned these models on traffic light data sets. We used two data sets: a) camera images captured in the simulator (~120 training images), and b) camera images of the test parking lot provided in the course ROS bag file (~300 training images). We hand labelled each image with bounding boxes of all the traffic lights, as well as the traffic light color associated with each bounding box using [labelImg](https://github.com/tzutalin/labelImg).

We tried several models: Faster RCNN, SSD Inception v2 and SSDLite MobileNet, all of which are based on pre-trained models provided in the Tensorflow model zoo. Faster RCNN has the longest inference time but highest accuracy, while SSDLite MobileNet stands on the opposite end. Taking into account both inference speed as well as accuracy, we ended up adopting the middle ground -- SSD Inception v2. In the team's different environments, the Single-Shot Multibox Detector model provides reasonably fast inference. A non-GPU laptop with Udacity's virtual machine running handled inference in about ~250 ms, while another better, GPU system running Linux natively delivered inferences in the single digit millisecond range. The VM and simulator performance has been a challenge during the project implementation. Extensive work has been put into finding, training, and evaluating potential models that deliver the required accuracy and speed.

In the simulator, the SSD model used has 100% accuracy on red light images, ~98% accuracy for green light, and unfortunately a low ~33% accuracy for yellow light. Because most of its mistakes are concentrated on yellow light images, and for those instances it mostly wrongly classifies yellow light as red, we think this model satisfies our need.

An analysis of the models have shown that overfitting is an issue. The models are suitable for the limited scenarios where they will be used. But with more time and in a real world project, this would be an area to focus on even more.

You can find the frozen graphs at `ros/src/tl_detector/model`. For model training, please see   `tl_light_detector_python3/train_model/`.


#### Debugging
We used two useful methods for debugging and iterative improvement.

a) *Visualization*  
Turn on visual debugging when running the simulator by:

```
roslaunch launch/styx.launch visualize:=true
```

It uses rviz to visualize pedal and throttle report in a dynamic plot, as well as camera images and waypoints from the cars perspective when running the simulator. Keep in mind though, that the visualization may be too heavy for some computers to handle when running the simulator at the same time.

Click image below for demo video:  
[![demo video](https://img.youtube.com/vi/Mdwih7xQpdU/0.jpg)](https://youtu.be/Mdwih7xQpdU)

See code in `ros/src/visualize/`.

b) *Evaluation datasets*  
Running the car through the entire loop in the simulator takes a long time, and there are long stretches of road without traffic lights. To speed up the iteration process, we created an exhaustive dataset of all 6 traffic intersections in the simulator. For each traffic light, when it is within sight of the car, we capture the camera images from the car; we run the car through the loop several times to capture all color states of all lights. This dataset with ~2000 images is close to exhaustive and is a substitute for the simulator for the purpose of evaluating the performance of the traffic light classifier.

There is also a dataset available that has been extracted from a sample bag file provided by Udacity. The video in that bag file shows how the real car approaches a traffic light in an environment similar to where our project will be tested on that car. This dataset has been used to evaluate the real world model.


#### Traffic Light Detector Integration Logic
After we trained the model, we integrated it into our codebase with some additional logic. First, because we know the location of traffic lights based on a map a priori, we turn on the traffic light detection classifier only when the car is within a certain distance from the next closest traffic light to minimize computation resource and latency. With our analysis, this is the distance where the classifier can start reliably predicting.

Next, based on offline evaluation data, we implemented the logic that we adopt the traffic light state when any of the two conditions are met: a) three consecutive frames are detected to have the same traffic light color, or b) the classifier returns a high score (>0.4), indicating higher confidence in the prediction. This logic guarantees perfect performance at each light intersection in the simulator.

See code in `ros/src/tl_detector/tl_detector.py`.


### Waypoint Update

This module is tasked with producing a list of waypoints for the car to follow. In other words, it determines the car's speed and trajectory.

We are provided with a pre-determined trajectory. We use a KD tree to locate the closest waypoints ahead of the car from the trajectory. The number of waypoints to look ahead is dynamic based on the speed: it has a minimal threshold (35 waypoints), and when the car is driving fast, it loads more waypoints.

Using the information published by the traffic light detector channel, we decelerate when we see yellow / red lights, and continue driving at the same speed when we see green lights or no lights.

See code at `ros/src/waypoint_updater/waypoint_updater.py`.

### Twist Controller

The module is tasked with proposing throttle, brake and steering values to execute the trajectory proposed by the waypoint update module. We use a provided PID controller to realize this. See `ros/src/twist_controller/twist_controller.py`.

---
**Original Udacity project README:**

This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the [instructions from term 2](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images
