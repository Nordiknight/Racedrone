import cv2
import depthai as dai
import apriltag
import numpy as np
import math
import os

# Known AprilTag size (in meters)
TAG_SIZE = 0.165  # Adjust to match your actual printed tag size

# Camera intrinsics for OAK-D Lite 640x480 (can be tuned via calibration)
# These values are approximate and may need tuning!
fx = 460*1.1 # focal length x 460, 368
fy = 460*1.1 # focal length y
cx = 320*1.1 # optical center x
cy = 240*1.1 # optical center y

camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
], dtype=np.float64)

dist_coeffs = np.zeros((4, 1))  # Assuming no distortion

# 3D points of tag corners in tag's coordinate system
object_points = np.array([
    [-TAG_SIZE/2,  TAG_SIZE/2, 0],
    [ TAG_SIZE/2,  TAG_SIZE/2, 0],
    [ TAG_SIZE/2, -TAG_SIZE/2, 0],
    [-TAG_SIZE/2, -TAG_SIZE/2, 0],
], dtype=np.float32)

# DepthAI pipeline
pipeline = dai.Pipeline()
cam = pipeline.createColorCamera()
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setPreviewSize(640, 480)

xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam.preview.link(xout.input)

# Start device
with dai.Device(pipeline) as device:
    queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    detector = apriltag.Detector()
    file_path = "/home/racedrone/Test/coords.txt"
    
    if (os.path.isfile(file_path)):
    	coords = open("coords.txt", "w")
    else:	
    	coords = open("coords.txt", "x")
    print("Press 'q' to quit.")
    while True:
        inFrame = queue.get()
        frame = inFrame.getCvFrame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray)

        for tag in tags:
            corners = tag.corners.astype(np.float32)

            success, rvec, tvec = cv2.solvePnP(
                object_points,
                corners,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
		

                # Draw axis
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, TAG_SIZE / 2)

                # Extract translation
                x, y, z = tvec.flatten()
                cv2.putText(frame, f"ID: {tag.tag_id}", (int(tag.center[0]), int(tag.center[1])-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(frame, f"Pos: x={x:.2f}m y={y:.2f}m z={z:.2f}m",
                            (int(tag.center[0]), int(tag.center[1])-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
               	coords.write(f"x: {x}, y: {y}, z: {z}\n")

                # Extract rotation in Euler angles
                rot_mat, _ = cv2.Rodrigues(rvec)
                sy = math.sqrt(rot_mat[0, 0]**2 + rot_mat[1, 0]**2)
                singular = sy < 1e-6
                if not singular:
                    x_angle = math.atan2(rot_mat[2, 1], rot_mat[2, 2])
                    y_angle = math.atan2(-rot_mat[2, 0], sy)
                    z_angle = math.atan2(rot_mat[1, 0], rot_mat[0, 0])
                else:
                    x_angle = math.atan2(-rot_mat[1, 2], rot_mat[1, 1])
                    y_angle = math.atan2(-rot_mat[2, 0], sy)
                    z_angle = 0
                coords.write(f"X-Angle: {x_angle}, Y-Angle: {y_angle}, Z-Angle: {z_angle}\n")

                cv2.putText(frame, f"Rot: x={math.degrees(x_angle):.1f} y={math.degrees(y_angle):.1f} z={math.degrees(z_angle):.1f}",
                            (int(tag.center[0]), int(tag.center[1]) + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow("AprilTag Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
