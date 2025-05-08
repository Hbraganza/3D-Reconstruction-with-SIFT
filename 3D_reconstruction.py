import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def reconstruct_3d(mtx, dist):
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    images = [cv2.imread(os.path.join(r"C:\Users\Sensing and Perception\navi_v1.0\bunny_racer\multiview-10-canon_t4i\images", f"{i:03d}.jpg"), 0) 
              for i in range(35)]
    keypoints, descriptors = [], []

    # Detect features
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        keypoints.append(kp)
        descriptors.append(des)

    # Initialize camera poses and points
    poses = [np.hstack((np.eye(3), np.zeros((3, 1))))]  # First camera at origin
    points_3d = []

# Main script before the loop
    bf = cv2.BFMatcher()
    matches = bf.match(descriptors[0], descriptors[1])
    matches = sorted(matches, key=lambda x: x.distance)
    pts1 = np.float32([keypoints[0][m.queryIdx].pt for m in matches])
    pts2 = np.float32([keypoints[1][m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(pts1, pts2, mtx, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, mtx, mask=mask)

    # Camera poses
    poses = [np.eye(3, 4)]  # Image 0 at origin
    poses.append(np.hstack((R, t)))  # Image 1 pose

    # Triangulate initial 3D points
    P1 = mtx @ poses[0]
    P2 = mtx @ poses[1]
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = points_4d[:3] / points_4d[3]  # Shape (3, M)

    # Store 3D points and keypoint indices in image 1 (inliers only)
    inlier_indices = np.where(mask.ravel() > 0)[0]
    prev_keypoint_indices = [matches[j].trainIdx for j in inlier_indices]  # Indices in image 1
    prev_points_3d = [points_3d[:, j] for j in inlier_indices]  # Corresponding 3D points
    points_3d_list = [points_3d]  # List to store 3D points per pair

    def reconstruct_3d(mtx, dist):
        global poses, points_3d_list, prev_keypoint_indices, prev_points_3d
        for i in range(1, len(images) - 1):
            # Match features between image i and i+1
            matches = bf.match(descriptors[i], descriptors[i + 1])
            matches = sorted(matches, key=lambda x: x.distance)
        
            # Get keypoint indices
            kp_indices_i = [m.queryIdx for m in matches]    # Image i
            kp_indices_i1 = [m.trainIdx for m in matches]  # Image i+1
        
            # Map keypoint indices in image i to 3D points
            mapping = {idx: j for j, idx in enumerate(prev_keypoint_indices)}
        
            # Collect 2D-3D correspondences for image i+1
            correspondences_2d = []
            correspondences_3d = []
            for m in range(len(matches)):
                if kp_indices_i[m] in mapping:
                    correspondences_2d.append(keypoints[i + 1][kp_indices_i1[m]].pt)
                    correspondences_3d.append(prev_points_3d[mapping[kp_indices_i[m]]])
        
            if len(correspondences_2d) < 4:
                raise ValueError(f"Not enough correspondences for PnP at image {i+1}")
        
        # Estimate pose with PnP
            correspondences_2d = np.array(correspondences_2d, dtype=np.float32)
            correspondences_3d = np.array(correspondences_3d, dtype=np.float32).T  # Shape (3, K)
            _, rvec, tvec, inliers = cv2.solvePnPRansac(correspondences_3d.T, correspondences_2d, mtx, dist)
            R, _ = cv2.Rodrigues(rvec)
            pose = np.hstack((R, tvec))
            poses.append(pose)
        
        # Triangulate new 3D points
            P1 = mtx @ poses[i]
            P2 = mtx @ poses[i + 1]
            pts1 = np.float32([keypoints[i][m.queryIdx].pt for m in matches])
            pts2 = np.float32([keypoints[i + 1][m.trainIdx].pt for m in matches])
            points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
            new_points_3d = points_4d[:3] / points_4d[3]
            points_3d_list.append(new_points_3d)
        
            # Update for next iteration
            prev_keypoint_indices = [m.trainIdx for m in matches]  # Indices in image i+1
            prev_points_3d = [new_points_3d[:, j] for j in range(new_points_3d.shape[1])]
    
        return points_3d_list

    # Call in main script
    points_3d = reconstruct_3d(mtx, dist)

    # # Incremental SfM for remaining images
    # for i in range(1, len(images) - 1):
    #     matches = bf.match(descriptors[i], descriptors[i + 1])
    #     matches = sorted(matches, key=lambda x: x.distance)
    #     pts1 = np.float32([keypoints[i][m.queryIdx].pt for m in matches])
    #     pts2 = np.float32([keypoints[i + 1][m.trainIdx].pt for m in matches])

    #     # Find 2D-3D correspondences (simplified: use previous points)
    #     prev_pts = pts1[mask[:len(pts1)] > 0]
    #     curr_pts = pts2[mask[:len(pts2)] > 0]
    #     points_prev_3d = points_3d[-1][:, mask[:len(pts1)] > 0]

    #     _, rvec, tvec, inliers = cv2.solvePnPRansac(points_prev_3d.T, curr_pts, mtx, dist)
    #     R, _ = cv2.Rodrigues(rvec)
    #     pose = np.hstack((R, tvec))
    #     poses.append(pose)

    #     P1 = mtx @ poses[i]
    #     P2 = mtx @ poses[i + 1]
    #     points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    #     points_3d.append(points_4d[:3] / points_4d[3])

    #return np.hstack(points_3d)
    return(points_3d)

def save_point_cloud(points_3d, filename):
    header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
end_header
""".format(points_3d.shape[1])
    with open(filename, 'w') as f:
        f.write(header)
        for i in range(points_3d.shape[1]):
            f.write(f"{points_3d[0, i]} {points_3d[1, i]} {points_3d[2, i]}\n")

def visualize_point_cloud(points_3d):
        """Display the reconstructed 3D point cloud."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_3d[0, :], points_3d[1, :], points_3d[2, :], c='b', marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title("Reconstructed 3D Point Cloud")
        plt.show()
    


# Main Execution
if __name__ == "__main__":
    # Task 2

    with np.load("calibration_data.npz") as data:
    # Extract the arrays (assuming names are 'mtx' and 'dist')
        mtx = data['mtx']
        dist = data['dist']

    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:", dist)

    # Tasks 4 & 5
    print("Reconstructing 3D model...")
    points_3d = reconstruct_3d(mtx, dist)

    print("Visualizing the 3D point cloud...")
    visualize_point_cloud(points_3d)
