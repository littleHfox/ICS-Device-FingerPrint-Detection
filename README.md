# ICS-Device-FingerPrint-Detection
**必须安装库：scapy, pandas, scikit-learn**
1. 数据提取和清洗  
python DataExtraction.py <PCAP_file> ExtractedData.csv  
设计提取了从100w到600w之间的500w条，原始数据文件大小随意，不会爆内存。提取的数量不建议过大，下一步很慢且实在过大会爆内存（500w大概220M）
2. 特征提取和整理  
python FeatrueExtraction.py <ExtractedData.csv> <ExtractedFeatrue.csv>  
速度取决于样本复杂度，样本量大时速度显著变慢，设计仅取200w条，跑了一整天
3. 倒转非目标网络发起的流方向  
python FormatFeatrue.py <ExtractedData.csv> <FormattedFeatrue.csv>  
4. 特征排序和整理(需要python版本大于等于3.10)  
python OrderFeatrue.py ExtractedFeatrue.csv <Output_Ordered_Featrue_csv> <function_num>
5. 归一化  
python Normalization.py <input_Ordered_Featrue_csv> <output_Normalized_Featrue_csv> <function_num>
6. 离散化时间特征  
python DFT.py <NormalizedFeatrue.csv> <dft_featrue.csv>
7. 训练模型和测试  
(1). 随机森林算法  
python RandomForest.py  
(2). RNN  
python NeuralNetwork.py