import pynvml
import time
import threading
import numpy as np
from collections import defaultdict

class GPUMonitor:
    """
    Low-overhead GPU monitoring class using pynvml for measuring:
    - Memory usage (current, min, max)
    - Power consumption (current, average, total energy)
    - GPU utilization
    
    The monitor runs in a separate thread to minimize impact on training.
    """
    
    def __init__(self, sampleInterval=0.5):
        """
        Initialize the GPU monitor.
        
        Args:
            sampleInterval (float): Sampling interval in seconds
        """
        self.sampleInterval = sampleInterval
        self.running = False
        self.thread = None

        # Initialize NVML
        pynvml.nvmlInit()

        # Determine GPUs to monitor
        self.gpu_ids = list(range(pynvml.nvmlDeviceGetCount()))
        self.handles = []
        self.device_names = []
        for gpu_id in self.gpu_ids:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            self.handles.append(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            self.device_names.append(name)
        
        
        # Metrics storage
        self.per_gpu_metrics = {gpu_id: defaultdict(list) for gpu_id in self.gpu_ids}
        self.metrics = defaultdict(list)
        self.start_time = None
        self.end_time = None
        

        
    def start(self):
        """Start GPU monitoring in a separate thread."""
        if self.running:
            return
            
        self.running = True
        self.start_time = time.time()

        for gpu_id in self.gpu_ids:
            self.per_gpu_metrics[gpu_id].clear()
        
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()

        
    def stop(self):
        """Stop GPU monitoring and calculate aggregate metrics."""
        if not self.running:
            return
            
        self.running = False
        self.end_time = time.time()
        if self.thread:
            self.thread.join()
            
        # Calculate derived metrics
        self._calculate_aggregate_metrics()
        
    def _monitor_loop(self):
        """Main monitoring loop that collects GPU metrics at regular intervals."""
        while self.running:
            # Reset aggregated metrics for this sample
            current_aggregated = {
                'memory_used_mb': 0,
                'memory_total_mb': 0,
                'power_watts': 0,
                'gpu_util_percent': 0,
                'temperature_c': 0,
                'sm_clock_mhz': 0
            }
            
            # Collect metrics for each GPU
            for i, gpu_id in enumerate(self.gpu_ids):
                handle = self.handles[i]
                try:
                    # Memory usage (in MB)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_used = mem_info.used / 1024 / 1024
                    memory_total = mem_info.total / 1024 / 1024
                    self.per_gpu_metrics[gpu_id]['memory_used_mb'].append(memory_used)
                    self.per_gpu_metrics[gpu_id]['memory_total_mb'].append(memory_total)
                    current_aggregated['memory_used_mb'] += memory_used
                    current_aggregated['memory_total_mb'] += memory_total
                    
                    # Power usage (in Watts)
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                        self.per_gpu_metrics[gpu_id]['power_watts'].append(power)
                        current_aggregated['power_watts'] += power
                    except pynvml.NVMLError:
                        self.per_gpu_metrics[gpu_id]['power_watts'].append(0)
                    
                    # GPU utilization (in %)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.per_gpu_metrics[gpu_id]['gpu_util_percent'].append(util.gpu)
                    current_aggregated['gpu_util_percent'] += util.gpu
                    
                    # Temperature (in °C)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    self.per_gpu_metrics[gpu_id]['temperature_c'].append(temp)
                    current_aggregated['temperature_c'] += temp
                    
                    # Clock speeds
                    try:
                        clock_sm = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                        self.per_gpu_metrics[gpu_id]['sm_clock_mhz'].append(clock_sm)
                        current_aggregated['sm_clock_mhz'] += clock_sm
                    except pynvml.NVMLError:
                        self.per_gpu_metrics[gpu_id]['sm_clock_mhz'].append(0)
                    
                except pynvml.NVMLError as e:
                    print(f"NVML Error for GPU {gpu_id}: {e}")
            
            # Add aggregated metrics to the main metrics dictionary
            for key, value in current_aggregated.items():
                if key in ['gpu_util_percent', 'temperature_c', 'sm_clock_mhz']:
                    # For these metrics, we want the average across GPUs
                    value = value / len(self.gpu_ids) if self.gpu_ids else 0
                self.metrics[key].append(value)
                
            time.sleep(self.sampleInterval)
    
    def _calculate_aggregate_metrics(self):
        """Calculate aggregate metrics once monitoring is stopped."""

       # Duration
        self.metrics['duration_seconds'] = self.end_time - self.start_time
        
        # Process aggregated metrics
        memory_used = np.array(self.metrics['memory_used_mb'])
        power_readings = np.array(self.metrics['power_watts'])
        util_readings = np.array(self.metrics['gpu_util_percent'])

        # Memory metrics
        self.metrics['memory_min_mb'] = np.min(memory_used) if len(memory_used) > 0 else 0
        self.metrics['memory_max_mb'] = np.max(memory_used) if len(memory_used) > 0 else 0
        self.metrics['memory_avg_mb'] = np.mean(memory_used) if len(memory_used) > 0 else 0
        
        # Power/Energy metrics
        self.metrics['power_min_watts'] = np.min(power_readings) if len(power_readings) > 0 else 0
        self.metrics['power_max_watts'] = np.max(power_readings) if len(power_readings) > 0 else 0
        self.metrics['power_avg_watts'] = np.mean(power_readings) if len(power_readings) > 0 else 0
        
        # Total energy in joules (W*s) and kWh
        if len(power_readings) > 0:
            # Approximate the integral of power over time
            energy_joules = np.sum(power_readings) * self.sampleInterval
            self.metrics['energy_joules'] = energy_joules
            self.metrics['energy_kwh'] = energy_joules / 3600000  # Convert J to kWh
        else:
            self.metrics['energy_joules'] = 0
            self.metrics['energy_kwh'] = 0
            
        # Average GPU utilization
        self.metrics['gpu_util_avg_percent'] = np.mean(util_readings) if len(util_readings) > 0 else 0
        
        # Calculate per-GPU metrics
        for gpu_id in self.gpu_ids:
            gpu_metrics = self.per_gpu_metrics[gpu_id]
            gpu_memory_used = np.array(gpu_metrics['memory_used_mb'])
            gpu_power_readings = np.array(gpu_metrics['power_watts'])
            gpu_util_readings = np.array(gpu_metrics['gpu_util_percent'])
            
            # Add per-GPU summary metrics to the main metrics dictionary
            prefix = f"gpu{gpu_id}_"
            self.metrics[f"{prefix}memory_min_mb"] = np.min(gpu_memory_used) if len(gpu_memory_used) > 0 else 0
            self.metrics[f"{prefix}memory_max_mb"] = np.max(gpu_memory_used) if len(gpu_memory_used) > 0 else 0
            self.metrics[f"{prefix}memory_avg_mb"] = np.mean(gpu_memory_used) if len(gpu_memory_used) > 0 else 0
            
            self.metrics[f"{prefix}power_min_watts"] = np.min(gpu_power_readings) if len(gpu_power_readings) > 0 else 0
            self.metrics[f"{prefix}power_max_watts"] = np.max(gpu_power_readings) if len(gpu_power_readings) > 0 else 0
            self.metrics[f"{prefix}power_avg_watts"] = np.mean(gpu_power_readings) if len(gpu_power_readings) > 0 else 0
            
            if len(gpu_power_readings) > 0:
                energy_joules = np.sum(gpu_power_readings) * self.sampleInterval
                self.metrics[f"{prefix}energy_joules"] = energy_joules
                self.metrics[f"{prefix}energy_kwh"] = energy_joules / 3600000
            else:
                self.metrics[f"{prefix}energy_joules"] = 0
                self.metrics[f"{prefix}energy_kwh"] = 0
                
            self.metrics[f"{prefix}gpu_util_avg_percent"] = np.mean(gpu_util_readings) if len(gpu_util_readings) > 0 else 0
    

    def getMetrics(self):
        """Get all collected metrics."""
        return dict(self.metrics)
    
    def getGPUMetrics(self, gpu_id):
        """Get metrics for a specific GPU."""
        if gpu_id not in self.gpu_ids:
            raise ValueError(f"GPU ID {gpu_id} not monitored")
        
        metrics = dict(self.per_gpu_metrics[gpu_id])
        
        # Add calculated metrics if monitoring has stopped
        if not self.running and self.end_time:
            prefix = f"gpu{gpu_id}_"
            for key, value in self.metrics.items():
                if key.startswith(prefix):
                    metrics[key[len(prefix):]] = value
        
        return metrics
    
    def logTB(self, writer, tag_prefix="GPU", step=None):
        """Log GPU metrics to TensorBoard."""
        metrics = self.getMetrics()
        
        # Log scalar metrics
        for key, value in metrics.items():
            # Skip list metrics
            if isinstance(value, list):
                continue
            writer.add_scalar(f"{tag_prefix}/{key}", value, step)
    
    def printSummary(self):
        """Print a summary of the GPU metrics."""
        metrics = self.getMetrics()
        
        print(f"\n===== Multi-GPU Monitoring Summary ({len(self.gpu_ids)} GPUs) =====")
        print(f"Duration: {metrics['duration_seconds']:.2f} seconds")
        print(f"Total Memory Usage: {metrics['memory_avg_mb']:.2f} MB avg, {metrics['memory_max_mb']:.2f} MB peak")
        print(f"Total Power Consumption: {metrics['power_avg_watts']:.2f} W avg, {metrics['power_max_watts']:.2f} W peak")
        print(f"Total Energy: {metrics['energy_joules']:.2f} J ({metrics['energy_kwh']:.6f} kWh)")
        print(f"Average GPU Utilization: {metrics['gpu_util_avg_percent']:.2f}%")
        
        # Per-GPU summary
        for i, gpu_id in enumerate(self.gpu_ids):
            print(f"\n--- GPU {gpu_id}: {self.device_names[i]} ---")
            prefix = f"gpu{gpu_id}_"
            print(f"Memory Usage: {metrics[f'{prefix}memory_avg_mb']:.2f} MB avg, {metrics[f'{prefix}memory_max_mb']:.2f} MB peak")
            print(f"Power Consumption: {metrics[f'{prefix}power_avg_watts']:.2f} W avg, {metrics[f'{prefix}power_max_watts']:.2f} W peak")
            print(f"Energy: {metrics[f'{prefix}energy_joules']:.2f} J ({metrics[f'{prefix}energy_kwh']:.6f} kWh)")
            print(f"GPU Utilization: {metrics[f'{prefix}gpu_util_avg_percent']:.2f}% avg")
        
        print("===============================================")
    
    def saveCSV(self, filename):
        """Save detailed metrics to a CSV file."""
        import csv
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Save aggregated time series data
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'memory_used_mb', 'power_watts', 'gpu_util_percent', 'temperature_c']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for i in range(len(self.metrics['memory_used_mb'])):
                timestamp = self.start_time + i * self.sampleInterval
                writer.writerow({
                    'timestamp': timestamp,
                    'memory_used_mb': self.metrics['memory_used_mb'][i],
                    'power_watts': self.metrics['power_watts'][i],
                    'gpu_util_percent': self.metrics['gpu_util_percent'][i],
                    'temperature_c': self.metrics['temperature_c'][i]
                })
        
        # Save per-GPU time series data
        for gpu_id in self.gpu_ids:
            gpu_filename = filename.replace('.csv', f'_gpu{gpu_id}.csv')
            
            with open(gpu_filename, 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'memory_used_mb', 'power_watts', 'gpu_util_percent', 'temperature_c']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                gpu_metrics = self.per_gpu_metrics[gpu_id]
                for i in range(len(gpu_metrics['memory_used_mb'])):
                    timestamp = self.start_time + i * self.sampleInterval
                    writer.writerow({
                        'timestamp': timestamp,
                        'memory_used_mb': gpu_metrics['memory_used_mb'][i],
                        'power_watts': gpu_metrics['power_watts'][i],
                        'gpu_util_percent': gpu_metrics['gpu_util_percent'][i],
                        'temperature_c': gpu_metrics['temperature_c'][i]
                    })
        
        # Save summary metrics
        summary_filename = filename.replace('.csv', '_summary.csv')
        metrics = self.getMetrics()
        
        with open(summary_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric', 'Value'])
            
            for key, value in metrics.items():
                if not isinstance(value, list):  # Skip the time series data
                    writer.writerow([key, value])
    
    def __del__(self):
        """Clean up NVML resources."""
        if self.running:
            self.stop()
        try:
            pynvml.nvmlShutdown()
        except:
            pass
