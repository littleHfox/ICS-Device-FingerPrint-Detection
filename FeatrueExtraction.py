import pandas as pd
import sys


def extract_featrue(input_csv, output_csv):
    df_in = pd.read_csv(input_csv)
    df_in.drop(['protocol_name'], axis=1, inplace=True)
    df_in['interval_time'] = None
    df_in['direction'] = None

    # 堆存储所有子流
    flows = []

    for index, pkt in df_in.iterrows():
        if index % 10000 == 0:
            print(f"{index}", end="\r", flush=True)
            if index == 2000000:
                break
        # 堆为空
        if len(flows) == 0:
            pkt['interval_time'] = 0
            pkt['direction'] = 1

            new_flow = [pkt]
            flows.append(new_flow)
            #print(f"New Flow append")
        # 堆不为空
        else:
            found = False
            for flow in reversed(flows):
                # 包属于已知子流
                if (pkt['src_ip'] == flow[0]['src_ip'] or pkt['src_ip'] == flow[0]['dst_ip']):
                    if(pkt['dst_ip'] == flow[0]['dst_ip'] or pkt['dst_ip'] == flow[0]['src_ip']):
                        if (pkt['src_port'] == flow[0]['src_port'] or pkt['src_port'] == flow[0]['dst_port']):
                            if (pkt['dst_port'] == flow[0]['dst_port'] or pkt['dst_port'] == flow[0]['src_port']):
                                if pkt['protocol'] == flow[0]['protocol']:                                  
                                    # 16个包为一个子流
                                    if len(flow) < 16:
                                        pkt['interval_time'] = pkt['timestamp'] - flow[-1]['timestamp']

                                        if pkt['src_ip'] == flow[0]['dst_ip']:
                                            pkt['direction'] = 0
                                        else:
                                            pkt['direction'] = 1

                                        flow.append(pkt)
                                        #print(f"pkt append")

                                    found = True
                                    break

            # 包不属于已知子流
            if not found:
                pkt['interval_time'] = 0
                pkt['direction'] = 1
                new_flow = [pkt]
                flows.append(new_flow)
                #print(f"New Flow append")

    
    # 处理完成，准备写入csv文件
    df_out = pd.DataFrame(columns=df_in.columns)
    empty_df = pd.DataFrame(None, index=[0], columns=df_in.columns)
    for flow in flows:
        # 不足16个包的子流都被舍弃
        if len(flow) == 16:
            new_df = pd.DataFrame(flow, columns=df_in.columns)

            df_out = pd.concat([df_out, new_df, empty_df], ignore_index=True)

    df_out.to_csv(output_csv, index=False)
    print(f"Extraction Complete. Data saved to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python FeatrueExtraction.py <input_csv> <output_csv>")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    extract_featrue(input_csv, output_csv)
