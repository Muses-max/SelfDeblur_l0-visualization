import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.websocket
import os,re
import time
import json,pickle,dill
import traceback
from tornado.options import define, options
from toy_process import _process
import multiprocessing
import psutil
import signal
import socket
import requests
from gpu_state_handler import get_gpu 
import platform
from pidhandler import pid2user
from multiprocessing import Process, Queue, Manager
from globaldict import globaldict
system = platform.platform()
islinux = system.lower().find("linux") >= 0
prefix = 'web'
port = 5000
port2 = 5001
validlist = ['vue.min.js', 'axios.min.js', 'axios.min.map', 'index.html', 'index.txt', 'bootstrap.min.css', 'bootstrap-switch.min.css', 'jquery.min.js','bootstrap-switch.min.js', 'bootstrap.min.css.map','bootstrap.min.js', 'bootstrap.css', 'bootstrap.min.css','bootstrap.css.map']
rootfile = '%s/%s'%(prefix, 'index.txt')
define("port", default=port, help="run on the given port", type=int)
SIGNAL_SIGKILL = 9
ip = socket.gethostbyname(socket.gethostname())
prefixstr = 'Single'


    
def writelog(content, path='log/log.txt', logtime=True, breakline=True, ip='', prefixstr=prefixstr):
    '''
    I think the logging module is difficult to use
    '''
    if logtime:
        if ip:
            content = prefixstr+ ' [' + time.strftime('%Y-%m-%d %H:%M:%S') + ', ip = %s ]'%ip + content
        else:
            content = prefixstr+' [' + time.strftime('%Y-%m-%d %H:%M:%S') + ']' + content
    if breakline:
        content += '\n'
    with open(path, 'a') as f:
        f.write(content)

        
def wait_child(signum, frame):
    '''
    Handling the zombie process
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
        data = data.replace('ip = "127.0.0.1"', 'ip="%s"'%ip)
        data = data.replace('port = 0', 'port=%s'%port)
        data = data.replace('port2 = 0', 'port2=%s'%port2)
    return data


def processwrapper(process, **kwargs):
    #print(kwargs)
    if not (kwargs.get('protocol') and kwargs.get('ip') and kwargs.get('port') and kwargs.get('handler')):
        return False
    url = '%s://%s:%s/%s'%(kwargs['protocol'], kwargs['ip'], kwargs['port'], kwargs['handler'])
    requests.post(url, json={'pid':os.getpid(), "file_name":__file__, "progress":0, "initial":"True", "finish":"False"})
    process(**kwargs)
    requests.post(url, json={'pid':os.getpid(), "file_name":__file__, "progress":-1, "initial":"False", "finish":"True"})
    
        
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
            valid = False
            if uri.find('dist/echarts') >= 0:
                path = uri
                valid = True
            else:
                path = os.path.join(prefix, uri)
            if (uri in validlist or valid) and os.path.exists(path):
                self.write(content(path))
                if path.find(".css") >= 0:
                    self.set_header("Content-Type","text/css");
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
        try:
            '''
            CORS
            '''
            self.set_header("Access-Control-Allow-Origin", "*") 
            self.set_header("Access-Control-Allow-Headers", "x-requested-with")
            self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
            state = self.get_argument("state")
            success = False 
            if state == "Start" and globaldict["process"] is None:
                p = multiprocessing.Process(target=processwrapper, args=(_process,), kwargs={"epochs":600, "protocol":"http", "ip":ip, "port":port2, "handler":"process"})
                p.start()
                globaldict["pid"] = p.pid
                globaldict["progress"]= 0
                globaldict["process"] = p
            else:
                if globaldict["pid"] > 0 and isrunning(globaldict["pid"]) and globaldict["process"] and globaldict["process"].is_alive():
                    globaldict["process"].terminate()
                    success = True
                    globaldict["process"] = None
                    globaldict["pid"] = -1
                    globaldict["progress"] = -1
            if state:
                self.write(json.dumps({"status":"ACK", "timestamp":time.strftime('%Y-%m-%d %H:%M:%S'), "error":"", "command":state, "success":str(success), "pid":globaldict["pid"]}))
            else:
                self.write(json.dumps({"status":"NAK", "timestamp":time.strftime('%Y-%m-%d %H:%M:%S'), "error":"state is None", "command":state, "success":"False", "pid":globaldict["pid"]}))
        except:
            writelog(traceback.format_exc(), ip=self.request.remote_ip)
            self.write("[%s NAK]\n"%time.strftime('%Y-%m-%d %H:%M:%S'))
        finally:
            writelog(json.dumps({"uri":self.request.uri, "method":self.request.method, "handler":self.__class__.__name__}), ip=self.request.remote_ip)
            
            
    def post(self):
        try:
            '''
            CORS
            '''
            self.set_header("Access-Control-Allow-Origin", "*") 
            self.set_header("Access-Control-Allow-Headers", "x-requested-with")
            self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
            data = json.loads(self.request.body.decode())
            state = data.get("state")
            success = False           
            if state == "Start" and globaldict["process"] is None:
                p = multiprocessing.Process(target=processwrapper, args=(_process,), kwargs={"epochs":600, "protocol":"http", "ip":ip, "port":port2, "handler":"process"})
                p.start()
                globaldict["pid"] = p.pid
                globaldict["progress"]= 0
                globaldict["process"] = p
            else:
                if globaldict["pid"] > 0 and isrunning(globaldict["pid"]) and globaldict["process"] and globaldict["process"].is_alive():
                    globaldict["process"].terminate()
                    success = True
                    globaldict["process"] = None
                    globaldict["pid"] = -1
                    globaldict["progress"] = -1
            if state:
                self.write(json.dumps({"status":"ACK", "timestamp":time.strftime('%Y-%m-%d %H:%M:%S'), "error":"", "command":state, "success":str(success), "pid":globaldict["pid"]}))
            else:
                self.write(json.dumps({"status":"NAK", "timestamp":time.strftime('%Y-%m-%d %H:%M:%S'), "error":"state is None", "command":state, "success":"False", "pid":globaldict["pid"]}))
        except:
            writelog(traceback.format_exc(), ip=self.request.remote_ip)
            self.write("[%s NAK]\n"%time.strftime('%Y-%m-%d %H:%M:%S'))
        finally:
            writelog(json.dumps({"uri":self.request.uri, "method":self.request.method, "handler":self.__class__.__name__}), ip=self.request.remote_ip)
            
            
    def options(self):
        self.set_header("Access-Control-Allow-Origin", "*") 
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
            
            
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
            globaldict["progress"] = progress
            jsondata = json.dumps({"pid":data.get("pid"), "progress":progress, "finish":str(data.get("finish")), "result":str(data.get("result"))})
            globaldict["websocketstring"] = jsondata
            WebSocketHandler.send_updates(jsondata)
            logjson = json.dumps({"mode":"IPC", "data":data, "method":self.request.method, "uri":self.request.uri, "handler":self.__class__.__name__, "process_path":tempfilename})
            if data.get("finish") == 'True':
                globaldict['pid'] = -1
                globaldict['progress'] = -1
                globaldict["process"] = None
            writelog(logjson, ip=self.request.remote_ip)
        except:
            writelog(traceback.format_exc(), ip=self.request.remote_ip)
            self.write("[%s NAK]\n"%time.strftime('%Y-%m-%d %H:%M:%S'))
        finally:
            writelog(json.dumps({"uri":self.request.uri, "method":self.request.method, "handler":self.__class__.__name__}), ip=self.request.remote_ip)

            
class WebSocketHandler(tornado.websocket.WebSocketHandler):
    cache = None
    
    def check_origin(self, origin):  
        return True 
    
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
                
                
                
class GPUHandler(tornado.websocket.WebSocketHandler):  
    @tornado.web.gen.coroutine
    def get(self):
        t = self.get_argument("type", "raw")
        adddiv = self.get_argument("adddiv", "True")
        self.action(t, adddiv=adddiv)
              
    def post(self):
        data = json.loads(self.request.body.decode())
        t = data.get("type", "raw")
        adddiv = data.get("adddiv", "True")
        self.action(t, adddiv=adddiv)
                
    def action(self, t, adddiv="True"):
        try:
            if t not in {"raw", "gpubars-view"}:
                t = "raw"
            data = get_gpu(t, adddiv=adddiv)
            self.write(json.dumps(data))
        except:
            writelog(traceback.format_exc(), ip=self.request.remote_ip)
        finally:
            writelog(json.dumps({"uri":self.request.uri, "method":self.request.method, "handler":self.__class__.__name__}), ip=self.request.remote_ip)
        
    
class PIDHandler(tornado.websocket.WebSocketHandler):
    def get(self):
        try:
            if islinux < 0:
                self.write("Sorry, this function is only supported on the Linux platform!") 
            self.action()
        except:
            writelog(traceback.format_exc(), ip=self.request.remote_ip)
            self.write(traceback.format_exc())
        finally:
            writelog(json.dumps({"uri":self.request.uri, "method":self.request.method, "handler":self.__class__.__name__}), ip=self.request.remote_ip)
            
    def action(self):
        pid = self.request.uri[1:]
        user, total = pid2user(pid)
        if user:
            self.write("PID %s belongs to %s:<br> %s"%(pid, user, total))
        else:
            self.write("PID %s does not exist!"%pid)
        
        
        
def main():
    tornado.options.parse_command_line()
    application = tornado.web.Application(
        [("/", MainHandler),("/raw", MainHandler),(r"/(?i)ping|/(?i)helloworld", HelloworldHandler),
        (r"/command", CommandHandler), (r"/process", ProcessHandler), ('/websocket', WebSocketHandler),(r"/.*\.(?i)html|/.*\.(?i)txt|/.*\.(?i)css|/.*\.(?i)js|/.*\.(?i)map", FileHandler), ('/gpu', GPUHandler), ('/gpus', GPUHandler), (r'/\d+', PIDHandler)]
    )
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(port2)
    http_server.start(num_processes=1)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
