import os
import requests
import json
import time

def _process(epochs=6000, protocol="http",ip="127.0.0.1", port=5000, handler="process"):
    if protocol == "https":
        ip = ip.replace("http", "https")
    url = '%s://%s:%s/%s'%(protocol, ip, port, handler)
    pid = os.getpid()
    count = 0
    for i in range(1,epochs+1):
        count += i
        if i % 600 == 0:
            requests.post(url, json={"pid":pid, "file_name":__file__, "progress":i, "initial":"False", "finish":"False", "result":str(count)})
        time.sleep(0.005)