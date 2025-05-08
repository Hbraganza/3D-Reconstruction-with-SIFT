# 3D-Reconstruction-with-SIFT
To use this software the following needs to be setup: <br/>

Python 3.8.20 installed.<br/>

After that the following libraries must be installed:<br/>

```
pip install glob
pip install open3d
pip install pyaml
pip install matplotlib
pip install opencv
```
These libraries are nessecary to run 3D_reconstruction.py<br/>

The following folders must be in the same directory as the 3D_reconstruction.py file: 
```
<object_images_folder> & <calibration images folder>
```

With this the python file 3D_reconstruction.py file will run and output it's result into the same directory. <br/>

To run planning.py ROS needs to be setup with noetic and moveit. The src folder needs to go in to your ROS workspace.<br/>

To run the code you must catkin_make in the workspace then in the same workspace source devel/setup.bash
after that is done then, run: 
```
roslaunch panda_moveit_config demo.launch
```
when it says it is ready open a new terminal.

change directory to the workspace.<br/>
run in the new terminal:
```
source devel/setup.bash
```
Then run:
```
rosrun panda_moveit_config planning.py
```
The planning.py file is found in src/panda_moveit_config/scripts


The 3d reconstruction uses the SIFT algorithm in Opencv. It was done by first calibrating the camera with the Opencv chessboard.
<br/>
![Image of chessboard calibration](https://github.com/user-attachments/assets/0688426e-43f4-4ce3-811a-ef27b9f12379)
<br/>
One photo of the calibration board used to make the data for the camera matrix/calibration matrix and distortion data.
<br/><br/>
Then, feature extraction is done on the object by looking at the vertices and edges in the image.
<br/>
![Feature_matching](https://github.com/user-attachments/assets/58b6571c-5bb3-47bc-8051-511f0f93a914)
<br/>
An example of the feature matching on the teddy bear between two different images with the lines showing the matching features detected.
<br/><br/>
This then produces an make a point cloud using those matching features which is output as an ply file, plus a .stl file using the poission meshing.
