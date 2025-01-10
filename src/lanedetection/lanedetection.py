import cv2
import numpy as np

import base64
import logging
import time
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.allMessages import mainCamera
import serial
import time
from src.hardware.serialhandler.threads.threadWrite import threadWrite

class LaneDetection:
    def __init__(self, queuesList, logger, debug=False):
        self.queuesList = queuesList
        self.debugger = debug
        self.logger = logger
        # self.logger = logging.getLogger("LaneDetection")
        self.serialCom = serial.Serial("/dev/ttyACM0", 115200, timeout=0.1)
        self.logFile = open('../logfile.log', 'a')
        tw = threadWrite(self.queuesList, self.serialCom, self.logFile, logging)
        
        # Subscribe to mainCamera messages
        self.mainCameraSubscriber = messageHandlerSubscriber(
            queuesList=self.queuesList,
            message=mainCamera, 
            deliveryMode="lastonly",
            subscribe=True
        )
    

    def decode_image(self, encoded_image):
        """Decodes a base64-encoded image."""
        decoded_bytes = base64.b64decode(encoded_image)
        np_array = np.frombuffer(decoded_bytes, np.uint8)
        return cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    def process_frame(self, frame):
        """Basic lane detection logic."""
        
        frame = cv2.resize(frame, (640, 480))
        height, width, _ = frame.shape
        
        # Convert to HSV for better white line detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for white color in HSV
        # Dynamically adjust brightness thresholds based on average frame brightness
        frame_brightness = np.mean(hsv[:,:,2])  # Get average brightness from V channel
        
        # Adjust thresholds based on ambient brightness
        if frame_brightness < 100:  # Dark environment
            lower_white = np.array([0, 0, 120])  # Lower threshold for dark conditions
            upper_white = np.array([180, 60, 255])
        elif frame_brightness > 200:  # Very bright environment
            lower_white = np.array([0, 0, 200])  # Higher threshold for bright conditions
            upper_white = np.array([180, 30, 255])
        else:  # Normal lighting
            lower_white = np.array([0, 0, 160])  # Medium threshold
            upper_white = np.array([180, 45, 255])
        
        # Create mask for white lines
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(white_mask, (5, 5), 0)
        
        # Apply Canny edge detection for lane edges
        edges = cv2.Canny(blurred, 50, 150)
        
        # Define region of interest (ROI)
        mask = np.zeros_like(edges)
        polygon = np.array([
            [(0, height), (width, height), (width // 2, height // 2)]
        ], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        cropped_edges = cv2.bitwise_and(edges, mask)

        # Hough Lines Transform
        lines = cv2.HoughLinesP(
            cropped_edges, rho=1, theta=np.pi/180, threshold=50,
            minLineLength=100, maxLineGap=70
        )

        return lines
    
    def calculate_lane_center(lines):
        if lines is not None and len(lines) > 0:
            x_coords = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                x_coords.extend([x1, x2])
            center_x = int(sum(x_coords) / len(x_coords))
            return center_x
        else:
            return None


    def run(self):
        """Main loop for processing frames."""
        while True:
            image_data = self.mainCameraSubscriber.receive()
            print("we are inside the run function")
            if image_data:
                try:
                    frame = self.decode_image(image_data)
                    processed_frame = self.process_frame(frame)
                    
                    centervalue=self.calculate_lane_center(processed_frame)
                    
                    if 250<=centervalue<=400:
                        command = {
                            "action": "vcd",
                            "speed": 20,
                            "steer": 0,
                            "time": 10
                        }
                        self.tw.sendToSerial(command)
                        print('centervalue', centervalue)
                        print("going straight*******") 
                    elif 250>centervalue:
                        command = {
                            "action": "vcd",
                            "speed": 10,
                            "steer": -15,
                            "time": 10
                        }
                        self.tw.sendToSerial(command)
                        print('centervalue', centervalue)
                        print("going left******") 
                    elif centervalue>400:
                        command = {
                            "action": "vcd",
                            "speed": 10,
                            "steer": 15,
                            "time": 10
                        }
                        self.tw.sendToSerial(command)
                        print('centervalue', centervalue)
                        print("going right******") 
                    else:
                        print("out of bounds********")
                    # return centervalue
                    # Optional: Save processed frame
                    # cv2.imwrite("output_frame.jpg", processed_frame)
                except Exception as e:
                    print("Error in lane detection:", e)
                    self.logger.error(f"Error in lane detection: {e}")
            else:
                if self.debugger:
                    self.logger.info("No new image data received.")