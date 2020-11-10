from multiprocessing import Process, Queue, Manager

manager = Manager()
globaldict = manager.dict()
globaldict["pid"] = -1 #pid = -1 means that the process is not running.
globaldict["progress"] = -1#'progress' means the progress of the subprocess. If progress = -1, the process is not running. If 'progress' is a non-negative integer, it indicates the current epoch of the subprocess.
globaldict["process"] = None
globaldict["websocketstring"] = "" #The string to be sent using websocket!
globaldict["ownerofws"] = None