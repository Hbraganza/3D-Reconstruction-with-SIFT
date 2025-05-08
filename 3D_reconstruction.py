#!/usr/bin/env python3

import cv2
import numpy as np
import glob
import open3d as o3d
import re
import matplotlib.pyplot as plt
import os


def camera_calibration(image_folder):
    """Calibrating the camera to get the intrinsic parameters focal length, scaling etc in the camera matrix and extrinsic parameter"""
    checkerboard_size = (9, 6)  # based on checkerboard
    square_size = 0.025  # Size of each square in meters

    # Prepare object points
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

    objpoints = []  # 3D points
    imgpoints = []  # 2D points

    images = glob.glob(os.path.join(image_folder, '*.jpg'))#taking the jpeg files from the correct directory
    if not images:
        raise ValueError("No calibration images found.")

    for fname in images:#going through each file individually to find the chessboard corners of each square
        img = cv2.imread(fname)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        if ret:
            objpoints.append(objp)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                             criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners_refined)

    if not objpoints:
        raise ValueError("No checkerboard corners detected. Check calibration images and parameters.")

    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist

def load_images(image_paths, K, dist_coeff):
# def load_images(image_paths):
    """ Load all images from specified paths as color images and apply undistortion. """
    images = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image at path {path} could not be loaded.")
        
        img = cv2.undistort(img, K, dist_coeff)
        img = cv2.edgePreservingFilter(img)
        images.append(img)
    
    return images

def feature_matching_and_display(img1, img2):
    """ Feature matching using SIFT and FLANN"""
    #Convert grayscale
    grays1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grays2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #Initialize SIFT detector
    sift = cv2.SIFT_create(nfeatures=19000, #number of features
                           contrastThreshold=0.02,
                           edgeThreshold=15)  #edge detection threshold the higher the less edge pieces are used

    # Find keypoints and descriptors
    keps1, descriptors_1 = sift.detectAndCompute(grays1, None)
    keps2, descriptors_2 = sift.detectAndCompute(grays2, None)

    # Initialize FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=1500)  # high number of checks = better matching

    flann = cv2.FlannBasedMatcher(index_params, search_params)#matching algorithm
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # Store all the good matches as per Lowe's ratio test.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # Relaxed Lowe's ratio test for more matches
            good_matches.append(m)

    print(f"Total matches found: {len(matches)}")
    print(f"Good matches after Lowe's ratio test: {len(good_matches)}")

    # Draw matches
    img_matches = cv2.drawMatches(img1,
                                  keps1,
                                  img2,
                                  keps2,
                                  good_matches,
                                  None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    #location of good matches
    src_pts = np.float32([keps1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([keps2[m.trainIdx].pt for m in good_matches])

    return src_pts, dst_pts, img_matches

def recover_camera_pose(src_pts, dst_pts, cmtx):
    """ Recovering the relative camera pose using the Essential Matrix and recoverpose"""
    # Compute the Essential Matrix using RANSAC
    E, mask = cv2.findEssentialMat(src_pts, dst_pts, cmtx, method=cv2.RANSAC, prob=0.99, threshold=2.0)#high probability threshold to reduce noise signifcantly

    # use Essential Matrix to obtain the relative rotation and translation
    _, R, t, mask_pose = cv2.recoverPose(E, src_pts, dst_pts, cmtx)

    return R, t, mask_pose

def triangulate_points(src_pts, dst_pts, Re1, t1, Re2, t2, cmtx):
    """ Triangulate 3D points from corresponding image points in the two views. Using the camera pose and feature matching"""
    # Projection matrices
    Pr1 = cmtx @ np.hstack((Re1, t1))
    Pr2 = cmtx @ np.hstack((Re2, t2))

    # transpose
    src_pts = src_pts.T  
    dst_pts = dst_pts.T

    # Triangulate points
    points_4d_hom = cv2.triangulatePoints(Pr1, Pr2, src_pts, dst_pts)#homogenous transform
    points_3d = points_4d_hom[:3, :] / points_4d_hom[3, :]  # Convert to Euclidean coordinates

    return points_3d

def bundle_adjustment(points_3d, src_pts, dst_pts, cmtx, R, t):
    """ Simple bundle adjustment to refine camera pose and 3D points and then put it back into triangulation """
    # Ensure points are in the correct format 
    object_points = points_3d.T.astype(np.float64)
    image_points = dst_pts.astype(np.float64)
    camera_matrix = cmtx.astype(np.float64)
    dist_coeffs = np.zeros((4, 1))  # Assume no distortion coefficients

    # Initialize rotational and translation vector
    if R is not None and t is not None:
        rvec, _ = cv2.Rodrigues(R)
        tvec = t.astype(np.float64)
    else:
        rvec = np.zeros((3, 1))
        tvec = np.zeros((3, 1))

    # Check if there are enough points for solvePnP
    if object_points.shape[0] < 4 or image_points.shape[0] < 4:
        print("Not enough points to run solvePnP")
        return None, None, None

    # Refine pose using solvePnP
    success, rvec, tvec = cv2.solvePnP(object_points, 
                                       image_points, 
                                       camera_matrix, 
                                       dist_coeffs, 
                                       rvec,
                                       tvec,
                                       useExtrinsicGuess=True,
                                       flags=cv2.SOLVEPNP_ITERATIVE)

    if not success:
        print("solvePnP failed to find a solution.")
        return None, None, None

    # Convert rotation vector back to rotation matrix
    R_refined, _ = cv2.Rodrigues(rvec)
    t_refined = tvec

    # Update 3D points using new pose
    Po2 = np.hstack((R_refined, t_refined))
    Po2 = cmtx @ Po2
    Po1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
    Po1 = cmtx @ Po1
    points_4d_hom = cv2.triangulatePoints(Po1, Po2, src_pts.T, dst_pts.T)
    points_3d_refined = points_4d_hom[:3, :] / points_4d_hom[3, :]

    return points_3d_refined, R_refined, t_refined

def visualize_point_cloud(points_3d):
    """ Visualize the point cloud and prep for .ply file"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Optionally orient normals to be consistent with the viewpoint
    pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))

    o3d.visualization.draw_geometries([pcd], window_name="3D Reconstruction", width=800, height=600, left=50, top=50)

def save_point_cloud_as_ply(points_3d, colors=None, filename="point_cloud_output.ply"):
    """ Save the point cloud as a PLY file for use in tools like Meshlab and gazebo to zoom in on"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(filename, pcd)
    print(f"Point cloud saved to {filename}")

def extract_colors_from_image(image, points):
    """Extract color values (BGR) from the image based on 2D points."""
    h, w = image.shape[:2]
    colors = []

    for point in points:
        x, y = int(point[0]), int(point[1])
        # Ensure the point is within image bounds
        if 0 <= x < w and 0 <= y < h:
            color = image[y, x]  # OpenCV uses (y, x) indexing
            colors.append(color / 255.0)  # Normalize to [0, 1] for Open3D
        else:
            colors.append([0, 0, 0])  # Black if point is out of bounds

    return np.array(colors)

def poisson_mesh_reconstruction(pcd, depth=9, density_threshold=0.01):
    """Generate mesh from point cloud using Poisson reconstruction."""
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth)
    
    # Clean up low-density vertices
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, density_threshold)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    mesh.compute_vertex_normals()
    mesh.orient_triangles()
    return mesh

def save_mesh(mesh, filename_base="output_mesh"):
    """Save mesh in formats compatible with Gazebo."""
    # Save in multiple formats
    o3d.io.write_triangle_mesh(f"{filename_base}.stl", mesh)  # For general use
    
    print(f"""Mesh saved as:
    - {filename_base}.stl: Standard STL format""")

def main():
    # File paths
# Load paths and sort by numerical value in filename
    image_paths = sorted(
    glob.glob('harry_teddy_v3/*.jpg'),
    key=lambda x: int(re.findall(r'\d+', x)[-1])  #To ensure you oreder the photos for no jumps
)
    calibration_path = 'Camera_matrix.txt'
    calibration_image_folder = 'Calibration_teddy'  # Checkerboard images

    if len(image_paths) < 2:
        print("At least two images are required.")
        return

    #Compute calibration data
    try:
        print("Calibrating camera...")
        K, dist_coeff = camera_calibration(calibration_image_folder)
        #save camera matrix to .txt
        np.savetxt(calibration_path, K, fmt='%.6f')
        print(f"Saved calibration data to {calibration_path}")

    except Exception as e:
        print(f"Calibration error: {e}")
        return

    print("Camera Matrix:\n", K)
    print("Distortion Coefficients:\n", dist_coeff)

    # Load images with undistortion
    try:
        images = load_images(image_paths, K, dist_coeff)
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    # Lists to hold 3D points, colors, and camera trajectory
    all_points_3d = []
    all_colors = []
    camera_trajectory = []

    # Initialize global rotation, translation
    R_total = np.eye(3)
    t_total = np.zeros((3, 1))
    camera_trajectory.append(t_total.flatten())

    # Store the previous pose
    previous_R = R_total.copy()
    previous_t = t_total.copy()

    for i in range(len(images) - 1):
        img1 = images[i]
        img2 = images[i + 1]

        print(f"Processing image pair {i + 1} and {i + 2}...")

        # Feature matching
        try:
            src_pts, dst_pts, img_matches = feature_matching_and_display(img1, img2)
            print("Feature matching done.")
        except ValueError as e:
            print(f"Feature matching error: {e}")
            continue

        # Recover pose
        try:
            R_rel, t_rel, mask_pose = recover_camera_pose(src_pts, dst_pts, K)
            print("Camera pose recovered.")
        except ValueError as e:
            print(f"Camera pose recovery error: {e}")
            continue

        # Normalize the translation to mitigate scale issues
        t_rel /= np.linalg.norm(t_rel)

        # Update global pose
        R_total = R_rel @ previous_R
        t_total = previous_t + previous_R @ t_rel

        # Triangulate
        try:
            points_3d = triangulate_points(src_pts, dst_pts,
                                           previous_R, previous_t,
                                           R_total, t_total, K)
            print("Triangulation completed.")
        except Exception as e:
            print(f"Triangulation error: {e}")
            continue

        # Optional Bundle Adjustment
        try:
            points_3d_refined, R_total, t_total = bundle_adjustment(points_3d, src_pts, dst_pts, K, R_total, t_total)
            if points_3d_refined is not None:
                points_3d = points_3d_refined
                print("Bundle adjustment completed.\n")
        except Exception as e:
            print(f"Bundle adjustment error: {e}")

        # Extract colors
        colors = extract_colors_from_image(img1, src_pts.reshape(-1, 2))

        all_points_3d.append(points_3d.T)
        all_colors.append(colors)

        # Update previous pose
        previous_R = R_total.copy()
        previous_t = t_total.copy()

        camera_trajectory.append(t_total.flatten())

    if not all_points_3d:
        print("No 3D points were reconstructed. Exiting.")
        return

    # Concatenate all 3D points and colors
    all_points_3d = np.concatenate(all_points_3d, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    print(f"Total 3D points reconstructed: {all_points_3d.shape[0]}")
    print(f"Point cloud shape: {all_points_3d.shape}")
    print(f"Color data shape: {all_colors.shape}\n")

    # Visualize the 3D point cloud
    visualize_point_cloud(all_points_3d)
    save_point_cloud_as_ply(all_points_3d)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points_3d)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    # Pre-process point cloud
    print("\nPre-processing point cloud...")
    pcd = pcd.voxel_down_sample(voxel_size=0.005)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.05, max_nn=50))
    
    #Poisson function
    print("\nRunning Poisson surface reconstruction...")
    try:
        poisson_mesh = poisson_mesh_reconstruction(pcd, depth=10, density_threshold=0.1)
        save_mesh(poisson_mesh, "poisson_mesh")
        o3d.visualization.draw_geometries([poisson_mesh], window_name="Poisson Mesh")
    except Exception as e:
        print(f"Poisson reconstruction failed: {str(e)}")
    
if __name__ == '__main__':
    main()
