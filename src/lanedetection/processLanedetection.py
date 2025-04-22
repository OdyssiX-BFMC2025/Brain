if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../../..")

from src.templates.workerprocess import WorkerProcess
from src.lanedetection.threads.threadLanedetection import threadLaneDetection

class processLaneDetection(WorkerProcess):
    """This process Lane Detection.\n
    Args:
            queueList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
            logging (logging object): Made for debugging.
            debugging (bool, optional): A flag for debugging. Defaults to False.
    """

    # ====================================== INIT ==========================================
    def __init__(self, queueList, logging, debugging=False):
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging
        super(processLaneDetection, self).__init__(self.queuesList)

    # ===================================== RUN ==========================================
    def run(self):
        """Apply the initializing methods and start the threads."""
        super(processLaneDetection, self).run()

    # ===================================== INIT TH ======================================
    def _init_threads(self):
        """Create the Lane Detection Publisher thread and add to the list of threads."""
        laneTh = threadLaneDetection(
         self.queuesList, self.logging, self.debugging
        )
        self.threads.append(laneTh)


# =================================== EXAMPLE =========================================
#             ++    THIS WILL RUN ONLY IF YOU RUN THE CODE FROM HERE  ++
#                  in terminal:    python3 processLanedetection.py
