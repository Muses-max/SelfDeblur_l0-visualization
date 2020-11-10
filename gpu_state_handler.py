import GPUtil
import re
import os
import traceback
from pidhandler import pid2user


def adapt(data, adddiv=True):
    data = re.sub(r'\n', '<br>', data)
    data = re.sub(r' ', '&nbsp', data)
    '''
    Consolas,Monaco,monospace: 等宽字体
    style= 这里无需加引号, 因为浏览器自己会加
    '''
    if adddiv:
        return '<div style=font-family:Consolas,Monaco,monospace>%s</div>'%data
    else:
        return data


def float_to_percentage(f,tostr=False):
    if tostr:
        return str(round(f*100))
    else:
        return round(f*100)


def parse(nvidia_output, gpucounts):
    s = ''
    l = nvidia_output.strip('\n').split('\n')
    lenl = len(l)
    start = -2 #从nvidia-smi的倒数第二行开始扫描
    d = dict()
    for cnt in range(gpucounts):
        d[cnt] = []
    while -start <= lenl:
        line = re.sub(r' +', ' ',l[start].strip('\n').strip('|').strip(' ')).split(' ')
        #print(line, len(line), line[0].isdigit(), line[1].isdigit())
        if len(line) == 5 and line[0].isdigit() and line[1].isdigit(): #GPU   PID   Type   [Process name]  [GPU Memory Usage]  
            d[int(line[0])].insert(0,(line[1], pid2user(line[1])[0]))
        else:
            break
        start -= 1
    for cnt in range(gpucounts):
        s += 'GPU %s<br>'%cnt
        for element in d[cnt]:
            s += '&nbsp&nbsp <a href="/%s">%s</a>&nbsp(%s)<br>'%(element[0], element[0], element[1])
    return s
    

def _raw(adaption=False,adddiv=True, linux=True):
    error = ""
    data = ""
    state = True
    gpucounts = len(GPUtil.getGPUs())
    try:
        with os.popen("nvidia-smi") as f:
            data = f.read()
        state = True
        pidinfo = parse(data, gpucounts)
        if adaption:
            data = adapt(data, adddiv=adddiv)
    except:
        error = 'Command [nvidia-smi] is not properly configured: <br>' + traceback.format_exc()
        pidinfo = ''
        if adaption:
            error = adapt(error, adddiv=False)
        state = False
    return {"state":str(state), "view":data, "error":error, "count":gpucounts, "pidinfo":pidinfo}



def _view():
    error = ""
    view = ""
    state = True
    gpudict = dict()
    GPUs = GPUtil.getGPUs()
    lengpus = len(GPUs)
    try:
        gpudict["view"] = dict()
        gpudict["view"]["y_data"] = list(map(str, range(lengpus)))[::-1]
        gpudict["view"]["Memory_Util"] = []
        gpudict["view"]["GPU_Util"] = []
        for i in range(lengpus-1, -1, -1): #echarts里面顺序是倒过来的
            gpudict["view"]["Memory_Util"].append({"value":float_to_percentage(-GPUs[i].memoryUtil)})
            gpudict["view"]["GPU_Util"].append({"value":float_to_percentage(GPUs[i].load)})
            gpudict["view"]["y_data"][-(i+1)] += ': '+ GPUs[i].name + ' (%s MB used/%s MB total)'%(GPUs[i].memoryUsed, GPUs[i].memoryTotal)
    except:
        error = adapt(traceback.format_exc())
        state = False
        gpudict["view"] = None
    finally:
        gpudict["error"] = error
        gpudict["state"] = str(state)
        gpudict["count"] = lengpus
    gpudict["users"] = ""
    gpudict["pidinfo"] = ""
    return gpudict


def get_gpu(t, adaption=True,adddiv="True",linux=True):
    adddiv = True if adddiv == "True" else False
    if t == 'raw':
        return _raw(adaption, adddiv=adddiv,linux=linux)
    elif t == 'view':
        return _view()
    else:
        raise Exception("What is it?")
    