import cv2
import numpy as np
import math

import base64
import time
import psutil, json, logging, inspect
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.allMessages import mainCamera, Record, serialCamera
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
        self.curr_steering_angle = 90
        # self.logger = logging.getLogger("LaneDetection")
        # self.serialCom = serial.Serial("/dev/ttyACM0", 115200, timeout=0.1)
        # self.logFile = open('../logfile.log', 'a')
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

    # subsciber function to recv images from the camera/serialCamera
    def subscribe(self):
        """Subscribe function. In this function we make all the required subscribe to process gateway"""
        for name, enum in self.messagesAndVals.items():
            subscriber = messageHandlerSubscriber(self.queueList, enum["enum"], "fifo", True)
            self.messages[name] = {"obj": subscriber}
        print("added a subscribtion to : ", self.messages.keys(), "for lane detection.")

                     
    def decode_image(self, encoded_image):
        """Decodes a base64-encoded image."""
        decoded_bytes = base64.b64decode(encoded_image)
        np_array = np.frombuffer(decoded_bytes, np.uint8)
        return cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    
    def detect_lane(self, frame):
        edges = self.detect_edges(frame)

        cropped_edges = self.region_of_interest(edges)

        line_segments = self.detect_line_segments(cropped_edges)
        lane_lines = self.average_slope_intercept(frame, line_segments)

        return lane_lines, frame 

    def detect_edges(self, frame):
        # filter for blue lane lines
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # show_image("hsv", hsv)
        lower_white = np.array([0, 0, 200], dtype=np.uint8)   # Low saturation, high brightness
        upper_white = np.array([180, 50, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_white, upper_white)
        # frame_brightness = np.mean(hsv[:,:,2])  # Get average brightness from V channel
        
        # Adjust thresholds based on ambient brightness
        # if frame_brightness < 100:  # Dark environment
        #     lower_white = np.array([0, 0, 120])  # Lower threshold for dark conditions
        #     upper_white = np.array([180, 60, 255])
        # elif frame_brightness > 200:  # Very bright environment
        #     lower_white = np.array([0, 0, 200])  # Higher threshold for bright conditions
        #     upper_white = np.array([180, 30, 255])
        # else:  # Normal lighting
        #     lower_white = np.array([0, 0, 160])  # Medium threshold
        #     upper_white = np.array([180, 45, 255])
        
        # Create mask for white lines
        # mask = cv2.inRange(hsv, lower_white, upper_white)
        # show_image("White Mask", mask)

        binary_mask = cv2.adaptiveThreshold(
            mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        # show_image("Thresholded Image", binary_mask)

        # detect edges
        # edges = cv2.Canny(mask, 200, 400)

        sobel_x = cv2.Sobel(binary_mask, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(binary_mask, cv2.CV_64F, 0, 1, ksize=5)
        edges = cv2.magnitude(sobel_x, sobel_y)
        edges = np.uint8(edges)
        # show_image("Sobel Edge Detection", edges)

        return edges
    
    def region_of_interest(self, canny):
        height, width = canny.shape
        mask = np.zeros_like(canny)

        # only focus bottom half of the screen
        polygon = np.array([[
            (0, height * 1 / 2),
            (width, height * 1 / 2),
            (width, height),
            (0, height),
        ]], np.int32)

        cv2.fillPoly(mask, polygon, 255)
        # show_image("mask", mask)
        masked_image = cv2.bitwise_and(canny, mask)
        return masked_image
    
    def detect_line_segments(self, cropped_edges):
        # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
        rho = 1  # precision in pixel, i.e. 1 pixel
        angle = np.pi / 180  # degree in radian, i.e. 1 degree
        min_threshold = 10  # minimal of votes
        line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), minLineLength=30, maxLineGap=4)

        # if line_segments is not None:
        #     for line_segment in line_segments:
        #         logging.debug('detected line_segment:')
        #         logging.debug("%s of length %s" % (line_segment, self.length_of_line_segment(line_segment[0])))

        return line_segments

    def make_points(self, frame, line):
        height, width, _ = frame.shape
        slope, intercept = line
        y1 = height  # bottom of the frame
        y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

        # bound the coordinates within the frame
        x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
        x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
        return [[x1, y1, x2, y2]]


    def average_slope_intercept(self, frame, line_segments):
        """
        This function combines line segments into one or two lane lines
        If all line slopes are < 0: then we only have detected left lane
        If all line slopes are > 0: then we only have detected right lane
        """
        lane_lines = []
        if line_segments is None:
            print('No line_segment segments detected')
            return lane_lines

        height, width, _ = frame.shape
        left_fit = []
        right_fit = []

        boundary = 1 / 3
        left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
        right_region_boundary = width * boundary  # right lane line segment should be on left 2/3 of the screen

        for line_segment in line_segments:
            for x1, y1, x2, y2 in line_segment:
                if x1 == x2:
                    print('skipping vertical line segment (slope=inf): ', line_segment)
                    continue
                fit = np.polyfit((x1, x2), (y1, y2), 1)
                slope = fit[0]
                intercept = fit[1]
                if slope < 0:
                    if x1 < left_region_boundary and x2 < left_region_boundary:
                        left_fit.append((slope, intercept))
                else:
                    if x1 > right_region_boundary and x2 > right_region_boundary:
                        right_fit.append((slope, intercept))

        left_fit_average = np.average(left_fit, axis=0)
        if len(left_fit) > 0:
            lane_lines.append(self.make_points(frame, left_fit_average))

        right_fit_average = np.average(right_fit, axis=0)
        if len(right_fit) > 0:
            lane_lines.append(self.make_points(frame, right_fit_average))

        return lane_lines
    
    def compute_steering_angle(self, frame, lane_lines):
        """ Find the steering angle based on lane line coordinate
            We assume that camera is calibrated to point to dead center
        """
        if len(lane_lines) == 0:
            print('No lane lines detected, do nothing')
            return -90

        height, width, _ = frame.shape
        if len(lane_lines) == 1:
            print('Only detected one lane line, just follow it. ', lane_lines[0])
            x1, _, x2, _ = lane_lines[0][0]
            x_offset = x2 - x1
        else:
            _, _, left_x2, _ = lane_lines[0][0]
            _, _, right_x2, _ = lane_lines[1][0]
            camera_mid_offset_percent = 0.00  # 0.0 means car pointing to center, -0.03: car is centered to left, +0.03 means car pointing to right
            mid = int(width / 2 * (1 + camera_mid_offset_percent))
            x_offset = (left_x2 + right_x2) / 2 - mid

        # find the steering angle, which is angle between navigation direction to end of center line
        y_offset = int(height / 2)

        angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
        angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
        steering_angle = angle_to_mid_deg + 90  # this is the steering angle needed by picar front wheel
        return steering_angle

    def stabilize_steering_angle(self, curr_steering_angle, new_steering_angle, num_of_lane_lines, max_angle_deviation_two_lines=5, max_angle_deviation_one_lane=1):
        """
        Using last steering angle to stabilize the steering angle
        This can be improved to use last N angles, etc
        if new angle is too different from current angle, only turn by max_angle_deviation degrees
        """
        if num_of_lane_lines == 2:
            # if both lane lines detected, then we can deviate more
            max_angle_deviation = max_angle_deviation_two_lines
        else:
            # if only one lane detected, don't deviate too much
            max_angle_deviation = max_angle_deviation_one_lane

        angle_deviation = new_steering_angle - curr_steering_angle
        if abs(angle_deviation) > max_angle_deviation:
            stabilized_steering_angle = int(curr_steering_angle
                                            + max_angle_deviation * angle_deviation / abs(angle_deviation))
        else:
            stabilized_steering_angle = new_steering_angle
        return stabilized_steering_angle
    
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
        
    def steer(self, frame, lane_lines):
        if len(lane_lines) == 0:
            print('No lane lines detected, nothing to do.')
            return frame

        new_steering_angle = self.compute_steering_angle(frame, lane_lines)
        self.curr_steering_angle = self.stabilize_steering_angle(self.curr_steering_angle, new_steering_angle, len(lane_lines))
        final_turn = self.curr_steering_angle - 90
        print('take a turn of: ', self.curr_steering_angle - 90, ' degrees', 'right' if final_turn > 0 else 'left')
        '''
        logic for car control based on steer angle.
        '''

    def run(self):
        """Main loop for processing frames."""
        # checking if we are able to recv images in here from the pipe.
        # print("turning on while loop of lane detection:")
        print("debug: message ", self.messages)
        if "serialCamera" in self.messages:
            print("debug: subcriber object from lane detection file ", self.messages["serialCamera"]["obj"])
        while self._running:
            image = self.messages["serialCamera"]["obj"].receive()
            if image is None:
                print("No image received.")
                continue
            
            image_data = base64.b64decode(image)
            img = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(img, cv2.IMREAD_COLOR)
            lane_lines, frame = self.detect_lane(image)
            self.steer(frame, lane_lines)
            time.sleep(1)
            
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
