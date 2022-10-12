from Classifier_test import KNN_pro
from Classifier_test import KNN
from QA_ReuseWater_KG import Data_Prepare
import random
from statistics import mean
'''

本文档用于测试KNNpro与KNN两文档中算法的可靠性
输入需求数据
从Data_Prepare中提取data_origin
然后用两种方法验证可靠性：
1、留出法hold-out，训练测试7：3进行10次随机划分，结果取平均值
2、留一交叉验证法LOOCV方法进行交叉验证，即（Leave-one-out cross-validation）：
只用一个数据作为测试集，其他的数据都作为训练集，并将此步骤重复N次（N为数据集的数据数量）

最后计算准确率、精确率、召回率、ROC、AUC等评价指标
计算每一个工艺代码的TP、FP、TN、FN，然后计算这个工艺代码对应的准确率、精确率、召回率、F-Measure
绘制ROC曲线需要得到分类器输出为正样本的概率计算方法，KNN需要重新定义一个计算概率的方法，所以ROC、AUC暂时不计算
TP 被预测为正类的正类，即预测正确的正类
FP 被预测为正类的负类
TN 被预测为负类的负类，即预测正确的负类
FN 被预测为负类的正类


'''

DataP = Data_Prepare.DataPrepare()
data_origin, data_dict = DataP.Data4ClassifierTree()
print('\n','data_dict'*20)
print(data_dict,'\n')
print('\n','data_origin'*20)
print(data_origin,'\n')
# 水质数据顺序： quantity、runtime、recovery、COD_in、CI_in、Hardness_in、ss_in'、
#               COD_out、CI_out、Hardness_out、ss_out、COD_emi、CI_emi、Hardness_emi、ss_emi


'''
留出法hold-out，训练测试7：3进行10次随机划分，结果取平均值
'''

T_num_o_all = 0 # 记录原始KNN正确数量总值
T_num_wd_all = 0 # 记录水质改进KNN正确数量总值
T_num_p_all = 0 # 记录逻辑判断与水质距离改进型KNN正确数量总值
T_num_o_ave = 0 # 记录原始KNN正确数量平均值
T_num_wd_ave = 0 # 记录水质改进KNN正确数量平均值
T_num_p_ave = 0 # 记录逻辑判断与水质距离改进型KNN正确数量平均值
code_list = list(data_dict.keys())

count = 0
accuracy_o =  []
accuracy_wd = []
accuracy_p = [] # 记录每次循环测试的准确率，每个元素为一个数值
precision_o = []
precision_wd =[]
precision_p = [] # 记录每次循环测试的,每个工艺的精确率，每个元素为一个字典
recall_o = []
recall_wd = []
recall_p = []  # 记录每次循环测试的,每个工艺的召回率，每个元素为一个字典


for x in range(0,10):
    TP_o = {}
    TP_wd = {}
    TP_p = {}
    FP_o = {}
    FP_wd = {}
    FP_p = {}
    TN_o = {}
    TN_wd = {}
    TN_p = {}
    FN_o = {}
    FN_wd = {}
    FN_p = {}  # 给三个算法的TP、FP、TN、FN赋初使值
    for i in code_list:
        TP_o[i] = TP_wd[i] = TP_p[i] = FP_o[i] = FP_wd[i] = FP_p[i] = \
        TN_o[i] = TN_wd[i] = TN_p[i] = FN_o[i] = FN_wd[i] = FN_p[i] = 0

    precision_dict_o = {}
    precision_dict_wd = {}
    precision_dict_p = {} # 定义这次循环的精确率字典
    recall_dict_o  = {}
    recall_dict_wd = {}
    recall_dict_p = {} # 定义这次循环的召回率字典
    for i in code_list:
        precision_dict_o[i] = precision_dict_wd[i] = precision_dict_p[i] =\
        recall_dict_o[i] = recall_dict_wd[i] = recall_dict_p[i] = 0
        # 选取测试集
    train_set_h = []
    for j in data_origin:
        train_set_h.append(j)
    len_h = len(data_origin)
    test_number = int(len_h*0.3) # 选取30%的测试集数量
    test_set_h = random.sample(data_origin,test_number)
    for i in test_set_h:
        train_set_h.remove(i)
    print('训练集:',train_set_h)
    print('测试集:',test_set_h)

    T_num_o = 0   # 记录原始型正确的数量
    T_num_wd = 0  # 记录水质距离改进型正确的数量
    T_num_p = 0   # 记录逻辑判断与水质距离改进型正确的数量

    for data in test_set_h:
        KNN_p = KNN_pro.KNN_pro_Class()
        KNN_o = KNN.KNN_Class()
        d = KNN_o.KNN_original(data, train_set_h)
        # print('\n', '原始型_originalKNN_' * 20)
        # print(d, '\n')
        e = KNN_o.KNN_wd(data, train_set_h)
        # print('\n', '水质距离改进型_wdKNN_' * 20)
        # print(e, '\n')
        a, b, c = KNN_p.KNN_pro_calculate(data, train_set_h)
        # print('\n', '逻辑判断与水质距离改进型_proKNN_' * 20)
        # print('最近项目ID：', a, '\n最近工艺code：', b, '\n最近项目名称：', c)
        '''
        TP 被预测为正类的正类，即预测正确的正类
        FP 被预测为正类的负类
        TN 被预测为负类的负类，即预测正确的负类
        FN 被预测为负类的正类
        '''
        for i in code_list:
            ## o算法
            if d == i: # 预测工艺为i的正类
                if i == data[-2]: # 样本工艺也为i的正类
                    T_num_o += 1
                    TP_o[i] += 1  # TP增加一个
                else:             # 样本工艺为i的负类
                    FP_o[i] += 1 # FP增加一个
            else:      # 预测工艺为i的负类
                if i == data[-2]:  # 样本工艺为i的正类
                    FN_o[i] += 1  # FN增加一个
                else:  # 样本工艺为i的负类
                    TN_o[i] += 1  # TN增加一个

            ## wd算法
            if e == i: # 预测工艺为i的正类
                if i == data[-2]: # 样本工艺也为i的正类
                    T_num_wd += 1
                    TP_wd[i] += 1  # TP增加一个
                else:             # 样本工艺为i的负类
                    FP_wd[i] += 1 # FP增加一个
            else:      # 预测工艺为i的负类
                if i == data[-2]:  # 样本工艺为i的正类
                    FN_wd[i] += 1  # FN增加一个
                else:  # 样本工艺为i的负类
                    TN_wd[i] += 1  # TN增加一个

            ## p算法
            if i in b: # 预测工艺为i的正类
                if i == data[-2]: # 样本工艺也为i的正类
                    T_num_p += 1
                    TP_p[i] += 1  # TP增加一个
                else:             # 样本工艺为i的负类
                    FP_p[i] += 1 # FP增加一个
            else:      # 预测工艺为i的负类
                if i == data[-2]:  # 样本工艺为i的正类
                    FN_p[i] += 1  # FN增加一个
                else:  # 样本工艺为i的负类
                    TN_p[i] += 1  # TN增加一个

    for i in code_list:
        if TP_o[i]+FP_o[i] != 0:
            precision_dict_o[i] = TP_o[i]/(TP_o[i]+FP_o[i])
        if TP_wd[i]+FP_wd[i] != 0:
            precision_dict_wd[i] = TP_wd[i]/(TP_wd[i]+FP_wd[i])
        if TP_p[i]+FP_p[i] != 0:
            precision_dict_p[i] = TP_p[i]/(TP_p[i]+FP_p[i])
        if TP_o[i]+FN_o[i] != 0:
            recall_dict_o[i] =  TP_o[i]/(TP_o[i]+FN_o[i])
        if TP_wd[i] + FN_wd[i] != 0:
            recall_dict_wd[i] = TP_wd[i]/(TP_wd[i]+FN_wd[i])
        if TP_p[i] + FN_p[i] != 0:
            recall_dict_p[i] = TP_p[i]/(TP_p[i]+FN_p[i])

    precision_o.append(precision_dict_o)
    precision_wd.append(precision_dict_wd)
    precision_p.append(precision_dict_p)
    recall_o.append(recall_dict_o)
    recall_wd.append(recall_dict_wd)
    recall_p.append(recall_dict_p)

    accuracy_o_1 = T_num_o / test_number
    accuracy_wd_1 = T_num_wd / test_number
    accuracy_p_1 = T_num_p / test_number
    accuracy_o.append(accuracy_o_1)
    accuracy_wd.append(accuracy_wd_1)
    accuracy_p.append(accuracy_p_1)



    T_num_o_all += T_num_o  # 记录原始KNN正确数量平均值
    T_num_wd_all += T_num_wd  # 记录水质改进KNN正确数量平均值
    T_num_p_all += T_num_p  # 记录逻辑判断与水质距离改进型KNN正确数量平均值
    T_num_o_ave = T_num_o_all/10  # 记录原始KNN正确数量平均值
    T_num_wd_ave = T_num_wd_all/10  # 记录水质改进KNN正确数量平均值
    T_num_p_ave = T_num_p_all/10 # 记录逻辑判断与水质距离改进型KNN正确数量平均值
    count += 1
    print('⭐'*30,'hold-out测试结束第', count, '次','⭐'*30,'\n\n')
print('hold-out测试结束：','\n','accuracy_o：',accuracy_o,'\n','accuracy_wd：',accuracy_wd,'\n','accuracy_p：',accuracy_p,'\n',
      'precision_o：',precision_o,'\n','precision_wd：',precision_wd,'\n','precision_p：',precision_p,'\n',
      'recall_o：',recall_o,'\n','recall_wd：',recall_wd,'\n','recall_p：',recall_p)
## 计算准确率的平均值
mean_accuracy_o = mean(accuracy_o)
mean_accuracy_wd = mean(accuracy_wd)
mean_accuracy_p = mean(accuracy_p)
## 计算每个工艺下的精确率、召回率的平均值
mean_precision_o_dict = {}
mean_precision_wd_dict = {}
mean_precision_p_dict = {}
mean_recall_o_dict = {}
mean_recall_wd_dict = {}
mean_recall_p_dict = {}

for i in code_list:
    a_mpo = []
    a_mpwd = []
    a_mpp = []
    a_mro = []
    a_mrwd = []
    a_mrp = []
    for j in precision_o:
        a_mpo.append(j[i])
        mean_precision_o_dict[i] = mean(a_mpo)
    for k in precision_wd:
        a_mpwd.append(k[i])
        mean_precision_wd_dict[i] = mean(a_mpwd)
    for l in precision_p:
        a_mpp.append(l[i])
        mean_precision_p_dict[i] = mean(a_mpp)
    for m in recall_o:
        a_mro.append(m[i])
        mean_recall_o_dict[i] = mean(a_mro)
    for n in recall_wd:
        a_mrwd.append(n[i])
        mean_recall_wd_dict[i] = mean(a_mrwd)
    for o in recall_p:
        a_mrp.append(o[i])
        mean_recall_p_dict[i] = mean(a_mrp)

# 计算平均的F-Measure
fmeasure_dict_o = {}
fmeasure_dict_wd = {}
fmeasure_dict_p = {}
for i in code_list:
    fmeasure_dict_o[i] = fmeasure_dict_wd[i] = fmeasure_dict_p[i] = 0

for i in code_list:
    if mean_precision_o_dict[i] + mean_recall_o_dict[i] != 0:
        fmeasure_dict_o[i] = 2 * mean_precision_o_dict[i] * mean_recall_o_dict[i] / (mean_precision_o_dict[i] + mean_recall_o_dict[i])
    if mean_precision_wd_dict[i] + mean_recall_wd_dict[i] != 0:
        fmeasure_dict_wd[i] = 2 * mean_precision_wd_dict[i] * mean_recall_wd_dict[i] / (mean_precision_wd_dict[i] + mean_recall_wd_dict[i])
    if mean_precision_p_dict[i] + mean_recall_p_dict[i] != 0:
        fmeasure_dict_p[i] = 2 * mean_precision_p_dict[i] * mean_recall_p_dict[i] / (mean_precision_p_dict[i] + mean_recall_p_dict[i])

print('\n','mean_accuracy_o：',mean_accuracy_o,'\n','mean_accuracy_wd：',mean_accuracy_wd,'\n','mean_accuracy_p：',mean_accuracy_p,'\n',
      'mean_precision_o_dict：',mean_precision_o_dict,'\n','mean_precision_wd_dict：',mean_precision_wd_dict,'\n','mean_precision_p_dict：',mean_precision_p_dict,'\n',
      'mean_recall_o_dict：',mean_recall_o_dict,'\n','mean_recall_wd_dict：',mean_recall_wd_dict,'\n','mean_recall_p_dict：',mean_recall_p_dict,'\n'
      'fmeasure_dict_o：',fmeasure_dict_o,'\n','fmeasure_dict_wd：',fmeasure_dict_wd,'\n','fmeasure_dict_p：',fmeasure_dict_p,'\n')
