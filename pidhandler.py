import os, re

def pid2user(pid):
    with os.popen("ps aux|grep %s"%pid) as f:
        data = f.readlines()
    for line in data:
        line2 = line
        line = line.strip('\n').strip(' ')
        line = re.sub(r' +', ' ', line)
        line = line.split(' ')
        if line[1].strip(' ') == pid:
            return line[0], line2
    return '', ''