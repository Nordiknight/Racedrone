import depthai as dai
import cv2
import apriltag
import numpy as np

# Create pipeline
pipeline = dai.Pipeline()

# --- MONO + STEREO SETUP ---
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
stereo.setSubpixel(True)  # 🆕 Subpixel accuracy
stereo.setConfidenceThreshold(230)  # 🆕 Increase confidence (0–255, higher = more accurate)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# --- RGB CAMERA ---
camRgb = pipeline.createColorCamera()
camRgb.setPreviewSize(640, 480)
camRgb.setInterleaved(False)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)

# --- SPATIAL LOCATION CALCULATOR ---
spatialLocationCalculator = pipeline.createSpatialLocationCalculator()
spatialLocationCalculator.setWaitForConfigInput(True)
spatialLocationCalculator.inputConfig.setBlocking(False)
stereo.depth.link(spatialLocationCalculator.inputDepth)

# --- OUTPUT STREAMS ---
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)

xoutSpatialData = pipeline.createXLinkOut()
xoutSpatialData.setStreamName("spatialData")
spatialLocationCalculator.out.link(xoutSpatialData.input)

xinSpatialCalcConfig = pipeline.createXLinkIn()
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

# --- RUN DEVICE ---
with dai.Device(pipeline) as device:
    rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    spatialCalcConfigIn = device.getInputQueue("spatialCalcConfig")

    detector = apriltag.Detector()

    while True:
        inRgb = rgbQueue.get()
        frame = inRgb.getCvFrame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        tags = detector.detect(gray)

        for tag in tags:
            cX, cY = tag.center.astype(int)

            # 🆕 Larger, safer ROI (20x20 pixels)
            roi_size = 10
            x1 = max(0, cX - roi_size) / 640
            y1 = max(0, cY - roi_size) / 480
            x2 = min(639, cX + roi_size) / 640
            y2 = min(479, cY + roi_size) / 480

            config = dai.SpatialLocationCalculatorConfigData()
            config.roi = dai.Rect(dai.Point2f(x1, y1), dai.Point2f(x2, y2))
            spatialCalcConfig = dai.SpatialLocationCalculatorConfig()
            spatialCalcConfig.addROI(config)
            spatialCalcConfigIn.send(spatialCalcConfig)

            # Wait for result
            spatialData = spatialCalcQueue.get()
            for sd in spatialData.getSpatialLocations():
                coords = sd.spatialCoordinates
                # Filter out invalid depth (Z = 0 or negative)
                if coords.z > 0:
                    print(f"[Tag ID {tag.tag_id}] X: {coords.x/1000:.2f}m Y: {coords.y/1000:.2f}m Z: {coords.z/1000:.2f}m")

                    # Draw result
                    cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)
                    cv2.putText(frame, f"ID:{tag.tag_id} Z:{coords.z/1000:.2f}m", (cX + 5, cY),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("AprilTag + Depth", frame)
        if cv2.waitKey(1) == ord('q'):
            break
