import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import os
import time

from tornado.options import define, options

prefix = 'web'
port = 5000
validlist = ['vue.min.js', 'axios.min.js', 'index.html', 'index.txt']
rootfile = '%s/%s'%(prefix, 'index.txt')
define("port", default=port, help="run on the given port", type=int)


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
    
class MainHandler(tornado.web.RequestHandler):
    def get(self):
        with open(rootfile) as f:
            self.write(f.read())

        
class HelloworldHandler(tornado.web.RequestHandler):
    def get(self):
        self.action()
        
    def post(self):
        self.action()
        
    def action(self):
        if self.request.uri.lower().find("ping") >= 0:
            self.write("pong")
        elif self.request.uri.lower().find("helloworld") >= 0:
            self.write("helloworld!")
            
            
class FileHandler(tornado.web.RequestHandler):
    def get(self):
        try:
            uri = self.request.uri.lower()[1:] #remove the first /
            path = os.path.join(prefix, uri)
            if uri in validlist and os.path.exists(path):
                with open(path) as f:
                    self.write(f.read())
            else:
                self.write("Invalid!")  
        except:
            pass
        
        
class CommandHandler(tornado.web.RequestHandler):
    def get(self):
         pass
                
    def post(self):
              
            
def main():
    tornado.options.parse_command_line()
    application = tornado.web.Application(
        [(r"/", MainHandler),(r"/(?i)ping|/(?i)helloworld", HelloworldHandler), (r"/.*\.(?i)html|/.*\.(?i)txt|/.*\.(?i)css|/.*\.(?i)js", FileHandler),
        (r"/command", CommandHandler)]
    )
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
