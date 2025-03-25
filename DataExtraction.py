import csv
import sys
from scapy.all import rdpcap, IP, TCP, UDP, RawPcapReader, Ether

def extract_pcap_info(pcap_file, output_csv):
    # 打开PCAP文件，比较慢还可能导致虚拟机死机
    # packets = rdpcap(pcap_file, count=5000000)
    # print("PCAP file read")
    
    with open(output_csv, 'w', newline='') as csvfile:
        # 提取五元组，包大小，TCP窗口大小
        fieldnames = ['timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'protocol_name', 'packet_size', 'tcp_window_size']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        with RawPcapReader(pcap_file) as pcap_reader:
            for i, (pkt_data, pkt_metadata) in enumerate(pcap_reader):
                if i % 100000 == 0:
                    print(i, end='\r', flush=True)
                if i < 0:
                    continue
                elif i > 2000000:
                    break

            #for pkt in packets:
                pkt = Ether(pkt_data)
                if IP in pkt:
                    src_ip = pkt[IP].src
                    dst_ip = pkt[IP].dst
                    proto = pkt[IP].proto
                    timestamp = pkt_metadata.sec + pkt_metadata.usec * 1e-6
                    packet_size = pkt_metadata.wirelen
                    src_port = None
                    dst_port = None
                    tcp_window_size = None
                    protocol_name = None

                    # 剔除非TCP和UDP报文
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
    
    print(f"Extraction Complete. Data saved to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python DataExtraction.py <input_pcap> <output_csv>")
        sys.exit(1)
    
    pcap_file = sys.argv[1]
    output_csv = sys.argv[2]
    extract_pcap_info(pcap_file, output_csv)
