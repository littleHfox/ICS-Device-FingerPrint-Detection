# ICS-Device-FingerPrint-Detection
**必须安装库：scapy, pandas, scikit-learn**
1. 数据提取和清洗  
python DataExtraction.py <PCAP_file> ExtractedData.csv  
这一步在打开pcap文件时会非常慢，取决于pcap文件的大小，500w条数据花费了10min，1kw被系统杀死了进程。。。
2. 特征提取和整理  
python FeatrueExtraction.py ExtractedData.csv ExtractedFeatrue.csv  
样本量大时速度显著变慢，超过100w时几乎不动了
3. 特征排序和整理(需要python版本大于等于3.10)  
python OrderFeatrue.py ExtractedFeatrue.csv <Output_Ordered_Featrue_csv> <function_num>