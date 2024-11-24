import psutil
import GPUtil

class MonitorResource:

    @staticmethod
    def monitor():
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        ram_usage = memory_info.used / (1024 ** 2)  # MB
        gpus = GPUtil.getGPUs()
        gpu_usage = gpus[0].load * 100 if gpus else 0
        return cpu_usage, ram_usage, gpu_usage
