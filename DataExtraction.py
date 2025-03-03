import csv
import sys
from scapy.all import rdpcap, IP, TCP, UDP

def extract_pcap_info(pcap_file, output_csv):
    packets = rdpcap(pcap_file)
    print("pcap readed")
    
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'protocol_name', 'packet_size', 'tcp_window_size']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for pkt in packets:
            if IP in pkt:
                src_ip = pkt[IP].src
                dst_ip = pkt[IP].dst
                proto = pkt[IP].proto
                timestamp = pkt.time
                packet_size = len(pkt)
                src_port = None
                dst_port = None
                tcp_window_size = None
                protocol_name = None

                if proto != 6 and proto != 17:
                    continue
                
                if TCP in pkt:
                    src_port = pkt[TCP].sport
                    dst_port = pkt[TCP].dport
                    tcp_window_size = pkt[TCP].window
                    protocol_name = "TCP"
                elif UDP in pkt:
                    src_port = pkt[UDP].sport
                    dst_port = pkt[UDP].dport
                    tcp_window_size = 0
                    protocol_name = "UDP"
                    
                writer.writerow({
                    'timestamp': timestamp,
                    'src_ip': src_ip,
                    'dst_ip': dst_ip,
                    'src_port': src_port,
                    'dst_port': dst_port,
                    'protocol': proto,
                    'protocol_name': protocol_name,
                    'packet_size': packet_size,
                    'tcp_window_size': tcp_window_size
                })
    
    print(f"Extraction completed. Data saved to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python DataExtraction.py <input_pcap> <output_csv>")
        sys.exit(1)
    
    pcap_file = sys.argv[1]
    output_csv = sys.argv[2]
    extract_pcap_info(pcap_file, output_csv)
