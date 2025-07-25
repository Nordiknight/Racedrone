import depthai as dai
import cv2
import apriltag
import numpy as np

# Create pipeline
pipeline = dai.Pipeline()

# Stereo depth setup
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# RGB camera
camRgb = pipeline.createColorCamera()
camRgb.setPreviewSize(640, 480)
camRgb.setInterleaved(False)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)

# Spatial location calculator
spatialLocationCalculator = pipeline.createSpatialLocationCalculator()
spatialLocationCalculator.setWaitForConfigInput(True)
spatialLocationCalculator.inputConfig.setBlocking(False)
stereo.depth.link(spatialLocationCalculator.inputDepth)

# Outputs
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)

xoutSpatialData = pipeline.createXLinkOut()
xoutSpatialData.setStreamName("spatialData")
spatialLocationCalculator.out.link(xoutSpatialData.input)

xinSpatialCalcConfig = pipeline.createXLinkIn()
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

# Start device
with dai.Device(pipeline) as device:
    rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    spatialCalcConfigIn = device.getInputQueue("spatialCalcConfig")
GroundContro
    detector = apriltag.Detector()

    while True:
        inRgb = rgbQueue.get()
        frame = inRgb.getCvFrame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        tags = detector.detect(gray)
        spatialLocations = []

        for tag in tags:
            (ptA, ptB, ptC, ptD) = tag.corners
            cX, cY = tag.center.astype(int)

            # Define ROI (around tag center)
            config = dai.SpatialLocationCalculatorConfigData()
            config.roi = dai.Rect(dai.Point2f((cX - 5)/640, (cY - 5)/480),
                                  dai.Point2f((cX + 5)/640, (cY + 5)/480))
            spatialCalcConfig = dai.SpatialLocationCalculatorConfig()
            spatialCalcConfig.addROI(config)
            spatialCalcConfigIn.send(spatialCalcConfig)

            spatialData = spatialCalcQueue.get()
            for sd in spatialData.getSpatialLocations():
                coords = sd.spatialCoordinates
                print(f"[AprilTag ID {tag.tag_id}] X: {coords.x/1000:.2f}m Y: {coords.y/1000:.2f}m Z: {coords.z/10:.2f}cm")

                # Draw box and info
                cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"ID:{tag.tag_id} Z:{coords.z/1000:.2f}m", (cX+5, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("AprilTag + Depth", frame)
        if cv2.waitKey(1) == ord('q'):
            break
