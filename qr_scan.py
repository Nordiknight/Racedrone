import depthai as dai
import cv2

# Create pipeline
pipeline = dai.Pipeline()
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(30)

xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam_rgb.preview.link(xout.input)

# Open device
with dai.Device(pipeline) as device:
    video = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    detector = cv2.QRCodeDetector()

    print("Press Ctrl+C to quit")
    while True:
        frame = video.get().getCvFrame()
        data, bbox, _ = detector.detectAndDecode(frame)

        if bbox is not None:
            # Draw bounding box
            for i in range(len(bbox)):
                pt1 = tuple(bbox[i][0].astype(int))
                pt2 = tuple(bbox[(i + 1) % len(bbox)][0].astype(int))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            if data:
                print(f"[QR] {data}")

        cv2.imshow("OAK-D QR Scanner", frame)
        if cv2.waitKey(1) == ord('q'):
            break
