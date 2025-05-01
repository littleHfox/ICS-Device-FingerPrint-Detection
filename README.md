# ICS-Device-FingerPrint-Detection
**必须安装库：scapy, pandas, scikit-learn, imbalanced-learn, torch**
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
A. 机器学习方法  
(1). 随机森林算法  
python RandomForest.py  
(2). 决策树算法  
python DecisionTree.py  
(3). KNN（K均值）算法  
python KNN.py  
(4). 梯度提升算法  
python GradientBoosting.py  
(5). 朴素贝叶斯算法（朴素高斯贝叶斯）  
python GaussianNB.py  
B. 神经网络方法  
(1). RNN（前馈神经网络）  
python RNN.py  
(2). CNN（卷积神经网络）  
python CNN.py  
(3). ResNet（残差神经网络）  
python ResNet_FNN.py (FNN风格的残差)  
python ResNet_CNN.py (CNN风格的残差)