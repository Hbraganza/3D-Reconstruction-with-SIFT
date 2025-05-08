# 3D-Reconstruction-with-SIFT
This is a 3d reconstruction of a teddy bear and uses the SIFT algorithm in Opencv. It was done by first calibrating the camera with the Opencv chessboard.
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
This then produces an make a point cloud using those matching features which is output as an stl file.
