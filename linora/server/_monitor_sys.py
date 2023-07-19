import time
import warnings

import psutil
import pandas as pd
from IPython.display import display_html, clear_output


__all__ = ['process_monitor']


def memory_transform(data):
    for r,i in enumerate(['KB', 'MB', 'GB', 'TB'], start=1):
        t = data/1024**r
        if t<1024:
            break
    return f"{round(t,1)}{i}"

def process_monitor(display=True, monitor_memory=True, monitor_gpu=True, monitor_disk=False, monitor_pid=False, 
                    sleep=5, sleep_total=None, process_min_bytes=1024**2):
    pynvml_state = False
    try:
        if monitor_gpu:
            import pynvml
            pynvml.nvmlInit()
            pynvml_state = True
    except:
        warnings.warn("should install 'pynvml' package.")
    
    sleep_count = 0
    while 1:
        try:
            process = {}
            if monitor_memory:
                virtual_memory = psutil.virtual_memory()
                process['memory_total'] = memory_transform(virtual_memory.total)
                process['memory_used'] = memory_transform(virtual_memory.used)
                process['memory_percent'] = f"{virtual_memory.percent}%"
                process['cpu_count_physics'] = psutil.cpu_count()
                process['cpu_count_logical'] = psutil.cpu_count(logical=False)

            if monitor_disk:
                process['disk'] = []
                for i in psutil.disk_partitions():
                    t = psutil.disk_usage(i.mountpoint)
                    process['disk'].append({'device':i.device, 'mountpoint':i.mountpoint, 
                                            'total':memory_transform(t.total), 'used':memory_transform(t.used), 
                                            'percent':f"{t.percent}%", 'fstype':i.fstype})

            if monitor_pid:
                process['pid'] = []
                for proc in psutil.process_iter():
                    try:
                        if proc.cmdline():
                            memory_used = proc.memory_full_info().uss
                            if memory_used>process_min_bytes:
                                data = proc.as_dict()
                                data = {i:data[i] for i in ['pid', 'ppid', 'username', 'name', 'create_time', 'memory_percent', 'cmdline']}
                                data['create_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data['create_time']))
                                data['memory_used'] = memory_transform(memory_used)
                                process['pid'].append(data)
                    except:
                        pass

            if monitor_gpu and pynvml_state:
                process['gpu_memory'] = []
                process['gpu_pid'] = []
                for i in range(pynvml.nvmlDeviceGetCount()):
                    data = {'Kernel':f'GPU:{i}'}
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    data['Name'] = pynvml.nvmlDeviceGetName(handle).decode()
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    data['Memory-Usage'] = f"{memory_transform(meminfo.used)} / {memory_transform(meminfo.total)}"
                    data['Temperature'] = f'{pynvml.nvmlDeviceGetTemperature(handle,0)}℃'
                    data['FanSpeed'] = f'{pynvml.nvmlDeviceGetFanSpeed(handle)}%'
                    process['gpu_memory'].append(data)

                    for j in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                        data = {'Kernel':f'GPU:{i}'}
                        data['PID'] = j.pid
                        data['Memory-Usage'] = f"{memory_transform(j.usedGpuMemory)}"
                        data['Process-name'] = pynvml.nvmlSystemGetProcessName(j.pid).decode()
                        process['gpu_pid'].append(data)

            if not display:
                return process
            clear_output(wait=True)
            if monitor_memory:
                temp = ['memory_total', 'memory_used', 'memory_percent', 'cpu_count_physics', 'cpu_count_logical']
                display_html(pd.DataFrame({i:[process[i]] for i in temp}))
            if monitor_gpu and pynvml_state:
                display_html(pd.DataFrame(process['gpu_memory']))
                display_html(pd.DataFrame(process['gpu_pid']))
            if monitor_disk:
                display_html(pd.DataFrame(process['disk']))
            if monitor_pid:
                display_html(pd.DataFrame(process['pid']))
        except:
            pass
        time.sleep(sleep)
        sleep_count += sleep
        if sleep_total is not None:
            if sleep_total>sleep_count:
                break

