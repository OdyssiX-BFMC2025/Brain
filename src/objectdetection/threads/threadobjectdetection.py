import base64
import time
import cv2
import numpy as np
from ultralytics import YOLO
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (mainCamera)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
import inspect
from src.utils.messages import allMessages
from enum import Enum
class threadobjectdetection(ThreadWithStop):
    """This thread handles objectdetection.
    Args:
        queueList (dictionary of multiprocessing.queues.Queue): Dictionary of queues where the ID is the type of messages.
        logging (logging object): Made for debugging.
        debugging (bool, optional): A flag for debugging. Defaults to False.
    """

    def __init__(self, queueList, logging, debugging=False):
        super(threadobjectdetection, self).__init__()
        self.queueList = queueList
        self.logging = logging
        self.debugging = debugging
        self.messages = {}
        self.messagesAndVals = {}

        self.getNamesAndVals()
        self.subscribe()
        self.model = YOLO("yolov8n.pt") 

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
        print("added a subscribtion to : ", self.messages.keys(), "for object detection.")
    
    def decode_image(self, encoded_image):
        """Decodes a base64-encoded image."""
        decoded_bytes = base64.b64decode(encoded_image)
        np_array = np.frombuffer(decoded_bytes, np.uint8)
        return cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    def run(self):
        print("object detection started*****************")
        while self._running:
            image = self.messages["serialCamera"]["obj"].receive()
            if image is None:
                print("No image received in object detection file.")
                continue
            frame = self.decode_image(image)
            
            results = self.model.predict(frame, stream=True, conf=0.6, imgsz=480)  # Lower resolution + streaming API
            for r in results:
                print("printing names of object*****")
                print(r.names)
            time.sleep(3)

