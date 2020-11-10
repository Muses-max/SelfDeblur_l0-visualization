import os,re
ports = [5000, 5001]
pids = set()
for port in ports:
    print("lsof -i:%s"%port)
    with os.popen("lsof -i:%s"%port) as f:
        lines = f.readlines()
        for line in lines:
            line = re.sub(r' +', r' ',line.strip('\n').strip(' '))
            line = line.split(' ')
            if len(line) >= 1 and line[1].isdigit():
                pids.add(line[1].strip())
            #print(pids)
for pid in pids:
    try:
        print(pid, "killer!")
        os.system("sudo kill -9 %s"%pid)
    except:
        pass
            
            