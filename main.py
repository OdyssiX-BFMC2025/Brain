# ===================================== GENERAL IMPORTS ==================================
import sys
import subprocess

import command_sender
from src.lanedetection.processLanedetection import processLaneDetection
from src.objectdetection.processobjectdetection import processobjectdetection



sys.path.append(".")
from multiprocessing import Queue, Event
import logging

logging.basicConfig(level=logging.INFO)

# ===================================== PROCESS IMPORTS ==================================

from src.gateway.processGateway import processGateway
from src.dashboard.processDashboard import processDashboard
from src.hardware.camera.processCamera import processCamera
from src.hardware.serialhandler.processSerialHandler import processSerialHandler
from src.data.Semaphores.Semaphores import processSemaphores
from src.data.TrafficCommunication.processTrafficCommunication import processTrafficCommunication
from src.utils.ipManager.IpReplacement import IPManager
from src.hardware.serialhandler.threads.threadWrite import threadWrite
import time
logFile = open('logfile.log', 'a')

# ======================================== SETTING UP ====================================
allProcesses = list()

queueList = {
    "Critical": Queue(),
    "Warning": Queue(),
    "General": Queue(),
    "Config": Queue(),
}

logging = logging.getLogger()

Dashboard = False
Camera = True
Semaphores = False
TrafficCommunication = False
SerialHandler = False

AutoStart = False
autolane = False
objectdetection = True

# ===================================== SETUP PROCESSES ==================================

# Initializing gateway
processGateway = processGateway(queueList, logging)
processGateway.start()

# Ip replacement
path = './src/dashboard/frontend/src/app/webSocket/web-socket.service.ts'
IpChanger = IPManager(path)
IpChanger.replace_ip_in_file()


# Initializing dashboard
if Dashboard:
    processDashboard = processDashboard( queueList, logging, debugging = False)
    allProcesses.append(processDashboard)

# Initializing camera
if Camera:
    processCamera = processCamera(queueList, logging , debugging = True)
    allProcesses.append(processCamera)

# Initializing semaphores
if Semaphores:
    processSemaphores = processSemaphores(queueList, logging, debugging = False)
    allProcesses.append(processSemaphores)

# Initializing GPS
if TrafficCommunication:
    processTrafficCommunication = processTrafficCommunication(queueList, logging, 3, debugging = False)
    allProcesses.append(processTrafficCommunication)

# Initializing serial connection NUCLEO - > PI
if SerialHandler:
    processSerialHandler = processSerialHandler(queueList, logging, debugging = True)
    allProcesses.append(processSerialHandler)

# AutoStart Engine
if AutoStart:
    # Instantiate the serial connection and the thread write handler
    command_sender.send_commands_continuously(queueList, logFile, logging)

# Lane detection module
if autolane:
    processLaneDetection = processLaneDetection(queueList, logging, debugging = False)
    allProcesses.append(processLaneDetection)

# Object detection module
if objectdetection:
    processobjectdetection = processobjectdetection(queueList, logging, debugging = False)
    allProcesses.append(processobjectdetection)


# ===================================== START PROCESSES ==================================
for process in allProcesses:
    process.daemon = True
    process.start()

# ===================================== DEBUG FOR QUEUELIST ===============================
print('')
print(" ********* debug : queueList: ", queueList)
print('')
# ===================================== STAYING ALIVE ====================================
blocker = Event()
try:
    blocker.wait()
except KeyboardInterrupt:
    print("\nCatching a KeyboardInterruption exception! Shutdown all processes.\n")
    for proc in reversed(allProcesses):
        print("Process stopped", proc)
        proc.stop()

processGateway.stop()
