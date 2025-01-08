import serial
import time
from src.hardware.serialhandler.threads.threadWrite import threadWrite

def send_commands_continuously(queueList, logFile, logging):
    # Initialize serial communication
    serialCom = serial.Serial("/dev/ttyACM0", 115200, timeout=0.1)
    tw = threadWrite(queueList, serialCom, logFile, logging)

    # kl to 30
    command = {
        "action": "kl",
        "mode": 30
    }
    tw.sendToSerial(command)

    # Define the command
    command = {
        "action": "vcd",
        "speed": 30,  # Example value for speed
        "steer": -20,  # Example value for steer
        "time": 100    # Example value for time
    }

    # Continuous loop to send the command
    try:
        while True:
            tw.sendToSerial(command)
            time.sleep(0.1)  # Adjust the interval between sending commands if needed
    except KeyboardInterrupt:
        print("Command sending stopped by user.")
    finally:
        serialCom.close()

# Ensure the script can be called directly if needed
if __name__ == "__main__":
    send_commands_continuously(queueList=None, logFile=None, logging=None)
