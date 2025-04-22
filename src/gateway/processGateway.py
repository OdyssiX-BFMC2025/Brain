#Process that handles the data distribution to the threads.
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import sys
from src.templates.workerprocess import WorkerProcess
# from ..templates.workerprocess import WorkerProcess``

from src.gateway.threads.threadGateway import threadGateway
# from ..gateway.threads.threadGateway import threadGateway


class processGateway(WorkerProcess):
    """This process handle all the data distribution\n
    Args:
        queueList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
        logger (logging object): Made for debugging.
        debugging (bool, optional): A flag for debugging. Defaults to False.
    """

    def __init__(self, queueList, logger, debugging=False):
        self.logger = logger
        self.debugging = debugging
        super(processGateway, self).__init__(queueList)

    # ===================================== RUN ===========================================
    def run(self):
        """Apply the initializing methods and start the threads."""

        super(processGateway, self).run()

    # ===================================== INIT TH ==========================================
    def _init_threads(self):
        """Initializes the gateway thread."""
        
        gatewayThread = threadGateway(self.queuesList, self.logger, self.debugging)
        self.threads.append(gatewayThread)


# =================================== EXAMPLE =========================================
#             ++    THIS WILL RUN ONLY IF YOU RUN THE CODE Owner HERE  ++
#                  in terminal:    python3 processGateway.py

if __name__ == "__main__":
    from multiprocessing import Pipe, Queue, Event
    import time
    import logging

    allProcesses = list()
    # We have a list of multiprocessing.Queue() which individualy represent a priority for processes.
    queueList = {
        "Critical": Queue(),
        "Warning": Queue(),
        "General": Queue(),
        "Config": Queue(),
    }
    logging = logging.getLogger()
    process = processGateway(queueList, logging, debugging=True)
    process.daemon = True
    process.start()

    pipeReceive1, pipeSend1 = Pipe()
    queueList["Config"].put(
        {
            "Subscribe/Unsubscribe": "suBsCribe",
            "Owner": "Camera",
            "msgID": 1,
            "To": {"receiver": 1, "pipe": pipeSend1},
        }
    )
    time.sleep(1)


    pipeReceive2, pipeSend2 = Pipe()
    queueList["Config"].put(
        {
            "Subscribe/Unsubscribe": "Subscribe",
            "Owner": "Camera",
            "msgID": 2,
            "To": {"receiver": 2, "pipe": pipeSend2},
        }
    )
    time.sleep(1)


    pipeReceive3, pipeSend3 = Pipe()
    queueList["Config"].put(
        {
            "Subscribe/Unsubscribe": "subscribe",
            "Owner": "Camera",
            "msgID": 3,
            "To": {"receiver": 3, "pipe": pipeSend3},
        }
    )
    time.sleep(1)

    print("all config messages sent.")

    
    # # Record the start time before sending messages
    # start_time_critical = time.time()
    # queueList["Critical"].put(
    #     {
    #         "Owner": "Camera",
    #         "msgID": 1,
    #         "msgType": "1111",
    #         "msgValue": "This is the text1",
    #     }
    # )
    # start_time_warning = time.time()
    # queueList["Warning"].put(
    #     {
    #         "Owner": "Camera",
    #         "msgID": 3,
    #         "msgType": "1111",
    #         "msgValue": "This is the text3",
    #     }
    # )
    # start_time_general = time.time()
    # queueList["General"].put(
    #     {
    #         "Owner": "Camera",
    #         "msgID": 2,
    #         "msgType": "1111",
    #         "msgValue": "This is the text2",
    #     }
    # )
    # print("all messages sent.")

    # # Measure the time taken for each message to be received
    # message_1 = pipeReceive1.recv()
    # latency_critical = time.time() - start_time_critical
    # print("message 1 received:", message_1, "latency:", latency_critical, "seconds")

    # message_2 = pipeReceive2.recv()
    # latency_general = time.time() - start_time_general
    # print("message 2 received:", message_2, "latency:", latency_general, "seconds")

    # message_3 = pipeReceive3.recv()
    # latency_warning = time.time() - start_time_warning
    # print("message 3 received:", message_3, "latency:", latency_warning, "seconds")

    # Sending 10,000 messages and measuring latency
    num_messages = 10
    start_times = {"Critical": [], "Warning": [], "General": []}
    latencies = {"Critical": [], "Warning": [], "General": []}

    for i in range(num_messages):
        # Record start time and send messages to each queue
        start_times["Critical"].append(time.time())
        queueList["Critical"].put(
            {
                "Owner": "Camera",
                "msgID": 1,
                "msgType": "1111",
                "msgValue": f"This is critical message {i + 1}",
            }
        )

        start_times["Warning"].append(time.time())
        queueList["Warning"].put(
            {
                "Owner": "Camera",
                "msgID": 2,
                "msgType": "1111",
                "msgValue": f"This is warning message {i + 1}",
            }
        )

        start_times["General"].append(time.time())
        queueList["General"].put(
            {
                "Owner": "Camera",
                "msgID": 3,
                "msgType": "1111",
                "msgValue": f"This is general message {i + 1}",
            }
        )

    print(f"All {num_messages} messages sent.")

    for i in range(num_messages):
        # Receive messages and calculate latency for each queue
        message_critical = pipeReceive1.recv()
        latency_critical = time.time() - start_times["Critical"][i]
        latencies["Critical"].append(latency_critical)

        message_warning = pipeReceive2.recv()
        latency_warning = time.time() - start_times["Warning"][i]
        latencies["Warning"].append(latency_warning)

        message_general = pipeReceive3.recv()
        latency_general = time.time() - start_times["General"][i]
        latencies["General"].append(latency_general)

    print(f"All {num_messages} messages received.")

    # Calculate and display average latency for each queue
    avg_latency_critical = sum(latencies["Critical"]) / num_messages
    avg_latency_warning = sum(latencies["Warning"]) / num_messages
    avg_latency_general = sum(latencies["General"]) / num_messages

    print(f"Average latency for Critical messages: {avg_latency_critical:.6f} seconds")
    print(f"Average latency for Warning messages: {avg_latency_warning:.6f} seconds")
    print(f"Average latency for General messages: {avg_latency_general:.6f} seconds")

    # ===================================== STAYING ALIVE ====================================

    process.stop()
