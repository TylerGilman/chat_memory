import psutil
import GPUtil
import time
from datetime import datetime

def monitor_resources():
    """Monitor system resources during training"""
    while True:
        try:
            # Get CPU and memory usage
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # Get GPU information
            gpus = GPUtil.getGPUs()
            
            # Print information
            print(f"\n=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
            print(f"CPU Usage: {cpu_percent}%")
            print(f"Memory Usage: {memory.percent}%")
            
            for gpu in gpus:
                print(f"\nGPU {gpu.id} ({gpu.name})")
                print(f"Memory: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB")
                print(f"Utilization: {gpu.load*100:.1f}%")
                print(f"Temperature: {gpu.temperature}°C")
            
            time.sleep(60)
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            time.sleep(60)

if __name__ == "__main__":
    monitor_resources()
