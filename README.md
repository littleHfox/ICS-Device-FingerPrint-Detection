# ICS-Device-FingerPrint-Detection
**必须安装库：scapy, pandas**
1. 数据提取和清洗  
python DataExtraction.py Modbus.pcap ExtractedData.csv
2. 特征提取和整理  
python FeatrueExtraction.py ExtractedData.csv ExtractedFeatrue.csv
3. 特征排序和整理(需要python版本大于等于3.10)  
python OrderFeatrue.py ExtractedFeatrue.csv <Output_Ordered_Featrue_csv> <function_num>