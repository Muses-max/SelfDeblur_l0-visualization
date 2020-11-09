import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.websocket
import os
import time
import json
import traceback
from tornado.options import define, options
from toy_process import _process
import multiprocessing
import psutil
import signal
import socket
import requests
prefix = 'web'
port = 5000
validlist = ['vue.min.js', 'axios.min.js', 'axios.min.map', 'index.html', 'index.txt']
rootfile = '%s/%s'%(prefix, 'index.txt')
define("port", default=port, help="run on the given port", type=int)
SIGNAL_SIGKILL = 9
ip = socket.gethostbyname(socket.gethostname())


    
def writelog(content, path='log/log.txt', logtime=True, breakline=True, ip=''):
    '''
    I think the logging module is difficult to use
    '''
    if logtime:
        if ip:
            content = '[' + time.strftime('%Y-%m-%d %H:%M:%S') + ', ip = %s ]'%ip + content
        else:
            content = '[' + time.strftime('%Y-%m-%d %H:%M:%S') + ']' + content
    if breakline:
        content += '\n'
    with open(path, 'a') as f:
        f.write(content)

        
def wait_child(signum, frame):
    '''
    To avoid the zombie process
    '''
    try:
        while True:
            childpid, status = os.waitpid(-1, os.WNOHANG)
            if childpid == 0:
                break
    except OSError as e:
        pass

            
signal.signal(signal.SIGCHLD, wait_child)


def isrunning(pid):
    if pid in psutil.pids():
        return True
    else:
        return False
    
    
def content(filename):
    with open(filename) as f:
        data = f.read()
    if filename.find("htm") >= 0 or filename.find("txt") >= 0:
        #Here I cannot use the template system of tornado, because {{}} conflicts with Vue.
        data = data.replace('ip:"127.0.0.1"', 'ip:"%s"'%ip)
        data = data.replace('port:0', 'port:%s'%port)
    return data


def processwrapper(process, **kwargs):
    #print(kwargs)
    if not (kwargs.get('protocol') and kwargs.get('ip') and kwargs.get('port') and kwargs.get('handler')):
        return False
    url = '%s://%s:%s/%s'%(kwargs['protocol'], kwargs['ip'], kwargs['port'], kwargs['handler'])
    requests.post(url, json={'pid':os.getpid(), "file_name":__file__, "progress":0, "initial":"True", "finish":"False"})
    process(**kwargs)
    requests.post(url, json={'pid':os.getpid(), "file_name":__file__, "progress":-1, "initial":"False", "finish":"True"})
    
    
        

class FSM:
    pid = -1 #pid = -1 means that the process is not running.
    progress = -1 #'progress' means the progress of the subprocess. If progress = -1, the process is not running. If 'progress' is a non-negative integer, it indicates the current epoch of the subprocess.
    global_process = {"process":None}
    
    
class MainHandler(tornado.web.RequestHandler):
    '''
    Return index.txt (or index.html)
    '''
    def get(self):
        with open(rootfile) as f:
            self.write(content(rootfile))

        
class HelloworldHandler(tornado.web.RequestHandler):
    '''
    Enable users to ping the server.
    '''
    def get(self):
        self.action()
        
    def post(self):
        self.action()
        
    def action(self, log=True):
        if self.request.uri.lower().find("ping") >= 0:
            self.write("pong")
        elif self.request.uri.lower().find("helloworld") >= 0:
            self.write("helloworld!")
        if log:
            writelog(json.dumps({"uri":self.request.uri, "method":self.request.method, "handler":self.__class__.__name__}), ip=self.request.remote_ip) 
            
            
class FileHandler(tornado.web.RequestHandler):
    '''
    Transfer files from backend to frontend, e.g. *.js, *.html, *.txt and *.css.
    '''
    def get(self):
        try:
            uri = self.request.uri.lower()[1:] #remove the first /
            path = os.path.join(prefix, uri)
            if uri in validlist and os.path.exists(path):
                self.write(content(path))
            else:
                self.write("Invalid!")  
        except:
            writelog(traceback.format_exc(), ip=self.request.remote_ip)
        finally:
            writelog(json.dumps({"uri":self.request.uri, "method":self.request.method,"handler":self.__class__.__name__}), ip=self.request.remote_ip)
            
        
        
class CommandHandler(tornado.web.RequestHandler):
    '''
    Receive the command from the frontend
    '''
    def get(self):
         pass
                
    def post(self):
        try:
            data = json.loads(self.request.body.decode())
            state = data.get("state")
            success = False            
            if state == "Start" and FSM.global_process["process"] is None:
                p = multiprocessing.Process(target=processwrapper, args=(_process,), kwargs={"epochs":600, "protocol":"http", "ip":ip, "port":port, "handler":"process"})
                p.start()
                FSM.pid = p.pid
                FSM.progress = 0
                FSM.global_process["process"] = p
            else:
                if FSM.pid > 0 and isrunning(FSM.pid) and FSM.global_process["process"] and FSM.global_process["process"].is_alive():
                    FSM.global_process["process"].terminate()
                    success = True
                    FSM.global_process["process"] = None
                    FSM.pid = -1
                    FSM.progress = -1
            if state:
                self.write(json.dumps({"status":"ACK", "timestamp":time.strftime('%Y-%m-%d %H:%M:%S'), "error":"", "command":state, "success":str(success), "pid":FSM.pid}))
            else:
                self.write(json.dumps({"status":"NAK", "timestamp":time.strftime('%Y-%m-%d %H:%M:%S'), "error":"state is None", "command":state, "success":"False", "pid":FSM.pid}))
        except:
            writelog(traceback.format_exc(), ip=self.request.remote_ip)
            self.write("[%s NAK]\n"%time.strftime('%Y-%m-%d %H:%M:%S'))
        finally:
            writelog(json.dumps({"uri":self.request.uri, "method":self.request.method, "handler":self.__class__.__name__}), ip=self.request.remote_ip)
            
            
class ProcessHandler(tornado.web.RequestHandler):
    '''
    This class implements the IPC (Inter-Process Communications).
    It receives the information from a process, e.g., the progress of the algorithm.
    '''
    def get(self):
        pass
    
    def post(self):
        try:
            data = json.loads(self.request.body.decode())
            (filepath,tempfilename) = os.path.split(data.get("file_name"))
            progress = int(data.get("progress"))
            FSM.progress = progress
            WebSocketHandler.send_updates(json.dumps({"pid":data.get("pid"), "progress":progress, "finish":str(data.get("finish")), "result":str(data.get("result"))}))
            logjson = json.dumps({"mode":"IPC", "data":data, "method":self.request.method, "uri":self.request.uri, "handler":self.__class__.__name__, "process_path":tempfilename})
            if data.get("finish") == 'True':
                FSM.pid = -1
                FSM.progress = -1
                FSM.global_process["process"] = None
            writelog(logjson, ip=self.request.remote_ip)
        except:
            writelog(traceback.format_exc(), ip=self.request.remote_ip)
            self.write("[%s NAK]\n"%time.strftime('%Y-%m-%d %H:%M:%S'))
        finally:
            writelog(json.dumps({"uri":self.request.uri, "method":self.request.method, "handler":self.__class__.__name__}), ip=self.request.remote_ip)

            
class WebSocketHandler(tornado.websocket.WebSocketHandler):
    cache = None
    
    def open(self):
        WebSocketHandler.cache = self
        
    def on_close(self):
        WebSocketHandler.cache = None
        
    @classmethod
    def send_updates(cls, chat=None):
        #print(WebSocketHandler.cache)
        if WebSocketHandler.cache:
            if not chat:
                WebSocketHandler.cache.write_message("Hello, world! from %s:%s"%(ip, port))
            else:
                WebSocketHandler.cache.write_message(chat)
    
        
def main():
    tornado.options.parse_command_line()
    application = tornado.web.Application(
        [(r"/", MainHandler),(r"/(?i)ping|/(?i)helloworld", HelloworldHandler), (r"/.*\.(?i)html|/.*\.(?i)txt|/.*\.(?i)css|/.*\.(?i)js|/.*\.(?i)map", FileHandler),
        (r"/command", CommandHandler), (r"/process", ProcessHandler), ('/websocket', WebSocketHandler)]
    )
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
