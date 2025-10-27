from scipy import stats
import numpy as np
from scipy.stats import ttest_rel

def data_check(data):
    # Check that the data satisfies a normal distribution
    # data type : list

    # 计算描述统计量
    mean = np.mean(data)
    std_dev = np.std(data)
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)

    # 输出描述统计量
    print("均值：", mean)
    print("标准差：", std_dev)
    print("偏度：", skewness)
    print("峰度：", kurtosis)

    # 执行 Shapiro-Wilk 检验
    shapiro_test_statistic, shapiro_p_value = stats.shapiro(data)

    # 输出 Shapiro-Wilk 检验结果
    print("Shapiro-Wilk 检验统计量：", shapiro_test_statistic)
    print("Shapiro-Wilk 检验 p 值：", shapiro_p_value)

    # 判断数据是否服从正态分布
    if shapiro_p_value < 0.05:
        print("数据不服从正态分布")
    else:
        print("数据服从正态分布")
    
def t_test(list_A, list_B):
    t_statistic, p_value = stats.ttest_ind(list_A, list_B)
    print("t statistic:", t_statistic)
    print("p value:", p_value)
    
def get_data(txt_file, line_begin, line_end):
    DSC = []
    HD = []
    ASD = []
    with open(txt_file, 'r') as file:
        lines = file.readlines()
    # 处理每一行
    for i in range(line_begin, line_end):
        dsc = lines[i].strip().split(':')[-3][:-3]
        # print(dsc)
        DSC.append(float(dsc))
        hd = lines[i].strip().split(':')[-2][:-3]
        HD.append(float(hd))
        asd = lines[i].strip().split(':')[-1]
        ASD.append(float(asd))
        # break
    return DSC, HD, ASD


if __name__ == "__main__":
    # 模型 A 的 DSC 指标
    
    path_o = '/home/hzy/projects/PENGWIN/Results_Task1_test/RepUXNET/RepUXNET_Jul29-10-02_cedl/log.txt'
    path_p = '/home/hzy/projects/PENGWIN/Results_Task1_test/RepUXNET/RepUXNET_Aug06-10-36_edgeLoss/log.txt'
    
    dsc_o, hd_o, asd_o = get_data(path_o, 0, 20)
    dsc_p, hd_p, asd_p = get_data(path_p, 0, 20)
    
    # 进行配对 t 检验
    print(" ===== Paired T-test ====")
    t_stat, p_value = ttest_rel(dsc_o, dsc_p)
    print(f"dsc t-statistic: {t_stat}, p-value: {p_value}")
    
    t_stat, p_value = ttest_rel(hd_o, hd_p)
    print(f"hd t-statistic: {t_stat}, p-value: {p_value}")
    
    t_stat, p_value = ttest_rel(asd_o, asd_p)
    print(f"asd t-statistic: {t_stat}, p-value: {p_value}")
    
    # 执行 Wilcoxon 符号秩和检验
    print(" ===== Wilcoxon signed-rank ====")
    statistic, p_value = stats.wilcoxon(dsc_o, dsc_p)
    print(f"dsc w-statistic: {statistic}, p-value: {p_value}")
    
    statistic, p_value = stats.wilcoxon(hd_o, hd_p)
    print(f"hd w-statistic: {statistic}, p-value: {p_value}")
    
    statistic, p_value = stats.wilcoxon(hd_o, hd_p)
    print(f"hd w-statistic: {statistic}, p-value: {p_value}")
    
    # 执行 Wilcoxon 符号秩和检验
    # t_statistic, p_value = stats.wilcoxon(dsc_our, dsc_vm)
    # print("***** wilcoxon *****")
    # print("Our vs VM")
    # print("DSC: t statistic:", t_statistic, "p value:", p_value)

    # t_statistic, p_value = stats.wilcoxon(hd_our, hd_vm)
    # print("HD95: t statistic:", t_statistic, "p value:", p_value)
    
    # t_statistic, p_value = stats.wilcoxon(asd_our, asd_vm)
    # print("ASD: t statistic:", t_statistic, "p value:", p_value)


    # 执行独立样本 t 检验
    # t_statistic, p_value = stats.ttest_ind(dsc_o, dsc_vm)
    # print("Our vs VM")
    # print("DSC: t statistic:", t_statistic, "p value:", p_value)
    
    # t_statistic, p_value = stats.ttest_ind(hd_o, hd_vm)
    # print("HD95: t statistic:", t_statistic, "p value:", p_value)
    
    # t_statistic, p_value = stats.ttest_ind(asd_o, asd_vm)
    # print("ASSD: t statistic:", t_statistic, "p value:", p_value)


    

    # 判断是否拒绝原假设（显著差异）
    # alpha = 0.05
    # if p_value < alpha:
    #     print("拒绝原假设，存在显著差异")
    # else:
    #     print("无法拒绝原假设，差异不显著")
    
    # data_check(hd_pcnet)
