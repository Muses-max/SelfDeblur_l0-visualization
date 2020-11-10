import GPUtil
import re
import os
import traceback


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
    
def _raw(adaption=False,adddiv=True):
    error = ""
    data = ""
    state = True
    try:
        with os.popen("nvidia-smi") as f:
            data = f.read()
        if adaption:
            data = adapt(data, adddiv=adddiv)
        state = True
    except:
        error = 'Command [nvidia-smi] is not properly configured: <br>' + traceback.format_exc()
        if adaption:
            error = adapt(error, adddiv=False)
        state = False
    return {"state":str(state), "view":data, "error":error}



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
    return gpudict


def get_gpu(t, adaption=True,adddiv="True"):
    adddiv = True if adddiv == "True" else False
    if t == 'raw':
        return _raw(adaption, adddiv=adddiv)
    elif t == 'view':
        return _view()
    else:
        raise Exception("What is it?")
    