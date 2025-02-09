import cv2
import numpy as np

import base64
import logging
import time
import psutil, json, logging, inspect, eventlet
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.allMessages import mainCamera, Record, serialCamera
import serial
from enum import Enum
import time
# from src.hardware.serialhandler.threads.threadWrite import threadWrite
from src.utils.messages import allMessages
from src.templates.threadwithstop import ThreadWithStop

class threadLaneDetection(ThreadWithStop):
    """Thread which detects lane."""

    def __init__(self, queueList, logger, debug=False):
        super(threadLaneDetection, self).__init__()
        self.queueList = queueList
        self.debugger = debug
        self.logger = logger
        # self.logger = logging.getLogger("LaneDetection")
        self.serialCom = serial.Serial("/dev/ttyACM0", 115200, timeout=0.1)
        self.logFile = open('../logfile.log', 'a')
        # tw = threadWrite(self.queueList, self.serialCom, self.logFile, logging)

        self.messages = {}
        self.messagesAndVals = {}

        self.getNamesAndVals()
        self.subscribe()

        # # Subscribe to mainCamera messages
        # self.mainCameraSubscriber = messageHandlerSubscriber(
        #     queueList=self.queueList,
        #     message=serialCamera, 
        #     deliveryMode="lastonly",
        #     subscribe=True
        # )

    def getNamesAndVals(self):
        """Extract all message names and values for processing."""
        classes = inspect.getmembers(allMessages, inspect.isclass)
        for name, cls in classes:
            if name == "serialCamera" and issubclass(cls, Enum):
                self.messagesAndVals[name] = {"enum": cls, "owner": cls.Owner.value}
        print("debug : messagesAndVals: ", self.messagesAndVals)

    def subscribe(self):
        """Subscribe function. In this function we make all the required subscribe to process gateway"""
        for name, enum in self.messagesAndVals.items():
            subscriber = messageHandlerSubscriber(self.queueList, enum["enum"], "fifo", True)
            self.messages[name] = {"obj": subscriber}
        print("added a subscribtion to : ", self.messages.keys(), "for lane detection.")

    # add a function that continuously recvs using the subscriber objects (inside message dict)
                     
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
        # checking if we are able to recv images in here from the pipe.
        print("turning on while loop of lane detection:")
        print("debug: message ", self.messages)
        if "serialCamera" in self.messages:
            print("debug: subcriber object from lane detection file ", self.messages["serialCamera"]["obj"])
        while self._running:
            if "serialCamera" in self.messages:
                # if self.messages["serialCamera"]["obj"].isDataInPipe():
                image = self.messages["serialCamera"]["obj"].receive()
                print("debug: image received from lane detection file ", image)
                # else:
                    # print("No data in the pipe from lane detection.")
            else:
                print("Key 'serialCamera' not found in messages. Current keys:", self.messages.keys())

            
            if image is not None:
                print("*****image is not None!!! -->   image: ", image)
                print("stopping for now")
                break
            else:
                print("image received None.")
                print('***** general queueList: ', self.queueList['General'])
                print('***** config queueList: ', self.queueList['Config'])
            time.sleep(2)  
    
        # while True:
            # image_data = self.mainCameraSubscriber.receive()
        # image_data = self.messages
        # print(image_data)
            # print("we are inside the run function", image_data)
            # if image_data:
            #     print("Looping \n")
            #     try:
            #         frame = self.decode_image(image_data)
            #         processed_frame = self.process_frame(frame)
                    
            #         centervalue=self.calculate_lane_center(processed_frame)
                    
            #         if 250<=centervalue<=400:
            #             command = {
            #                 "action": "vcd",
            #                 "speed": 20,
            #                 "steer": 0,
            #                 "time": 10
            #             }
            #             self.tw.sendToSerial(command)
            #             print('centervalue', centervalue)
            #             print("going straight*******") 
            #         elif 250>centervalue:
            #             command = {
            #                 "action": "vcd",
            #                 "speed": 10,
            #                 "steer": -15,
            #                 "time": 10
            #             }
            #             self.tw.sendToSerial(command)
            #             print('centervalue', centervalue)
            #             print("going left******") 
            #         elif centervalue>400:
            #             command = {
            #                 "action": "vcd",
            #                 "speed": 10,
            #                 "steer": 15,
            #                 "time": 10
            #             }
            #             self.tw.sendToSerial(command)
            #             print('centervalue', centervalue)
            #             print("going right******") 
            #         else:
            #             print("out of bounds********")
            #         # return centervalue
            #         # Optional: Save processed frame
            #         # cv2.imwrite("output_frame.jpg", processed_frame)
            #     except Exception as e:
            #         print("Error in lane detection:", e)
            #         self.logger.error(f"Error in lane detection: {e}")
            # else:
            #     print("No new image data received.")
            #     self.logger.info("No new image data received.")
            #     time.sleep(0.1)

    # =============================== START ===============================================
    def start(self):
        super(threadLaneDetection, self).start()

    # =============================== STOP ================================================
    def stop(self):
        super(threadLaneDetection, self).stop()
