import pandas as pd
import sys

# 排序提取间隔时间
def order_interval_time(input_csv, output_csv):
    df_in = pd.read_csv(input_csv)
    # 构造表格
    df_out = pd.DataFrame(columns=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'time0', 'time1', 'time2', 'time3', 'time4', 'time5', 'time6', 'time7'])

    i = 0
    row = None
    for index, pkt in df_in.iterrows():
        if index % 1000 == 0:
            print(f"{index}")

        # 跳过空行
        if pkt.isnull().all():
            continue
        # 逐行组装
        else:
            if i == 0:
                row = pd.Series(index=df_out.columns, dtype=object)
                row[['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']] = [
                pkt['src_ip'], pkt['dst_ip'], pkt['src_port'], pkt['dst_port'], pkt['protocol']
            ]
                row[f'time{i}'] = pkt['interval_time']
                i += 1
            elif i == 7:
                row[f'time{i}'] = pkt['interval_time']
                df_out.loc[len(df_out)] = row
                i = 0
            else:
                row[f'time{i}'] = pkt['interval_time']
                i += 1

    df_out.to_csv(output_csv, index=False)

# 排序提取包大小
def order_pkt_size(input_csv, output_csv):
    df_in = pd.read_csv(input_csv)
    df_out = pd.DataFrame(columns=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'packet_size0', 'packet_size1', 'packet_size2', 'packet_size3', 'packet_size4', 'packet_size5', 'packet_size6', 'packet_size7'])

    i = 0
    row = None
    for index, pkt in df_in.iterrows():
        if index % 1000 == 0:
            print(f"{index}")

        if pkt.isnull().all():
            continue
        else:
            if i == 0:
                row = pd.Series(index=df_out.columns, dtype=object)
                row[['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']] = [
                pkt['src_ip'], pkt['dst_ip'], pkt['src_port'], pkt['dst_port'], pkt['protocol']
            ]
                row[f'packet_size{i}'] = pkt['packet_size']
                i += 1
            elif i == 7:
                row[f'packet_size{i}'] = pkt['packet_size']
                df_out.loc[len(df_out)] = row
                i = 0
            else:
                row[f'packet_size{i}'] = pkt['packet_size']
                i += 1

    df_out.to_csv(output_csv, index=False)

# 排序提取TCP窗口大小
def order_tcp_window(input_csv, output_csv):
    df_in = pd.read_csv(input_csv)
    df_out = pd.DataFrame(columns=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'tcp_window0', 'tcp_window1', 'tcp_window2', 'tcp_window3', 'tcp_window4', 'tcp_window5', 'tcp_window6', 'tcp_window7'])

    i = 0
    row = None
    for index, pkt in df_in.iterrows():
        if index % 1000 == 0:
            print(f"{index}")

        if pkt.isnull().all():
            continue
        else:
            if i == 0:
                row = pd.Series(index=df_out.columns, dtype=object)
                row[['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']] = [
                pkt['src_ip'], pkt['dst_ip'], pkt['src_port'], pkt['dst_port'], pkt['protocol']
            ]
                row[f'tcp_window{i}'] = pkt['tcp_window_size']
                i += 1
            elif i == 7:
                row[f'tcp_window{i}'] = pkt['tcp_window_size']
                df_out.loc[len(df_out)] = row
                i = 0
            else:
                row[f'tcp_window{i}'] = pkt['tcp_window_size']
                i += 1

    df_out.to_csv(output_csv, index=False)

# 排序提取包方向
def order_direction(input_csv, output_csv):
    df_in = pd.read_csv(input_csv)
    df_out = pd.DataFrame(columns=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'direction0', 'direction1', 'direction2', 'direction3', 'direction4', 'direction5', 'direction6', 'direction7'])

    i = 0
    row = None
    for index, pkt in df_in.iterrows():
        if index % 1000 == 0:
            print(f"{index}")

        if pkt.isnull().all():
            continue
        else:
            if i == 0:
                row = pd.Series(index=df_out.columns, dtype=object)
                row[['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']] = [
                pkt['src_ip'], pkt['dst_ip'], pkt['src_port'], pkt['dst_port'], pkt['protocol']
            ]
                row[f'direction{i}'] = pkt['direction']
                i += 1
            elif i == 7:
                row[f'direction{i}'] = pkt['direction']
                df_out.loc[len(df_out)] = row
                i = 0
            else:
                row[f'direction{i}'] = pkt['direction']
                i += 1

    df_out.to_csv(output_csv, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python FeatrueExtraction.py <input_csv> <output_csv> <function_num>")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    function_num = int(sys.argv[3])

    match function_num:
        case 1:
            order_interval_time(input_csv, output_csv)
        case 2:
            order_pkt_size(input_csv, output_csv)
        case 3:
            order_tcp_window(input_csv, output_csv)
        case 4:
            order_direction(input_csv, output_csv)