import csv
import random

def generate_traffic(filename="network_traffic.csv", rows=300):
    headers = ["duration", "src_bytes", "dst_bytes", "total_packets", "same_srv_rate", "diff_srv_rate", "dst_host_count", "label"]
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for i in range(rows):
            # 5% chance of anomaly
            is_anomaly = random.random() < 0.05
            
            if not is_anomaly:
                # Normal Traffic
                duration = round(random.uniform(0.1, 0.5), 2)
                src_bytes = random.randint(180, 250)
                dst_bytes = random.randint(400, 550)
                packets = random.randint(4, 8)
                same_srv = round(random.uniform(0.8, 1.0), 2)
                diff_srv = round(random.uniform(0.0, 0.2), 2)
                host_count = random.randint(2, 10)
                label = 0
            else:
                # Anomaly Patterns (DoS, Probe, etc.)
                attack_type = random.choice(['dos', 'probe', 'transfer'])
                
                if attack_type == 'dos':
                    # DoS: High rate, low duration, high count
                    duration = 0.0
                    src_bytes = 0
                    dst_bytes = 0
                    packets = random.randint(1, 3)
                    same_srv = 0.0
                    diff_srv = 1.0
                    host_count = random.randint(150, 255)
                elif attack_type == 'probe':
                    # Probe: Scanning many ports
                    duration = round(random.uniform(0.1, 2.0), 2)
                    src_bytes = random.randint(0, 50)
                    dst_bytes = 0
                    packets = random.randint(1, 5)
                    same_srv = round(random.uniform(0.1, 0.3), 2)
                    diff_srv = round(random.uniform(0.6, 0.9), 2)
                    host_count = random.randint(20, 100)
                else:
                    # Data Exfiltration: Large transfer
                    duration = round(random.uniform(5.0, 20.0), 2)
                    src_bytes = random.randint(5000, 20000)
                    dst_bytes = random.randint(100, 500)
                    packets = random.randint(50, 200)
                    same_srv = round(random.uniform(0.9, 1.0), 2)
                    diff_srv = 0.0
                    host_count = random.randint(1, 5)

                label = 1
            
            writer.writerow([duration, src_bytes, dst_bytes, packets, same_srv, diff_srv, host_count, label])

if __name__ == "__main__":
    generate_traffic()
    print("Generated 300 rows of realistic network traffic data.")
