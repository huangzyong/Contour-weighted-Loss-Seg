import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def generate_color_wheel(num_colors):
    colors = []
    for i in range(num_colors):
        # 计算当前颜色的HSV色相值
        hue = i / num_colors
        # 将HSV色相值转换为RGB颜色
        rgb_color = mcolors.hsv_to_rgb([hue, 1, 1])
        colors.append(rgb_color)
    return colors

# 生成14种颜色的色相环
color_wheel_15 = generate_color_wheel(15)

# 生成15种颜色的色相环
color_wheel_15 = generate_color_wheel(15)

# AMOS
# 器官名称和对应的体积
organs = ['spleen', 'right kidney', 'left kidney', 'gallbladder', 'esophagus', 'liver', 'stomach', 'aorta', 'inferior vena cava', 'portal', 'splenic', 'veins', 'pancreas', 'right adrenal gland', 'left adrenal gland']
# percentages = [67634,47528,50246,10014,4995,437787,104945,37865,222623,23596,1059,1209,18240,42970,17324]
percentages_amos = [437787,104945,67634,50246,47528,42971,37865,23596,22623,18240,17324,10014,4995,1291,1059]
# 绘制饼图
plt.figure()  # 可以调整图形大小
plt.subplot(221)
# plt.pie(sorted(percentages_amos), labels=organs, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab20.colors)
plt.pie(sorted(percentages_amos), labels=None, autopct=None, startangle=90, colors=plt.cm.tab20.colors)
plt.axis('equal')  # 确保饼图是圆形的
# 添加标题
# plt.title('Organ Percentage Distribution')


# BraTS
tissues = ['WT', 'TC', 'ET']
percentages_brats = [95967, 35753, 21447]  #95966 0.1490 0.6275 0.223
colors = ["#EBBFBF", "#C0CAD9", "#F6DCC2"]
plt.subplot(222)
# plt.pie(sorted(percentages_brats), labels=tissues, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab20.colors)
plt.pie(sorted(percentages_brats), labels=None, autopct=None, startangle=90, colors=plt.cm.tab20.colors)
plt.axis('equal')  # 确保饼图是圆形的

# FLARE
# tissues = ['1', '2', '3', '4']
# percentages_brats = [1206954, 277388, 185138, 65680]
# colors = ["#EBBFBF", "#C0CAD9", "#F6DCC2", "#F6DCD2"]
# plt.subplot(223)
# # plt.pie(sorted(percentages_brats), labels=tissues, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab20.colors)
# plt.pie(sorted(percentages_brats), labels=None, autopct=None, startangle=90, colors=plt.cm.tab20.colors)
# plt.axis('equal')  # 确保饼图是圆形的
# plt.subplot(223)
# x = np.linspace(1, 14, 14)
# num_cls = [100, 45, 7, 1, 100, 66, 36, 11, 1, 1, 100, 64, 34, 9]
# # plt.figure(figsize=(10,8))
# x = np.linspace(1, 4, 4)
# num_cls_1 = [100, 45, 7, 1]
# plt.plot(x, num_cls_1, c='#ff7f0e', ls='-', linewidth=1.2, marker='*', ms=8, alpha=0.5, label='sacrum fragment')
# x = np.linspace(5, 10, 6)
# num_cls_1 = [100, 66, 36, 11, 1, 1]
# plt.plot(x, num_cls_1, c='#2ca02c', ls='-', linewidth=1.2, marker='d', ms=8, alpha=0.5, label='left hipbone fragment')
# x = np.linspace(11, 14, 4)
# num_cls_1 = [100, 64, 34, 9]
# plt.plot(x, num_cls_1, c='#d62728', ls='-', linewidth=1.2, marker='p', ms=8, alpha=0.5, label='right hipbone fragment')
# plt.rcParams.update({'font.size': 8}) 

# plt.legend()
# plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.5)
# plt.xticks(np.arange(0, 16, 1), fontsize=8)
# plt.yticks(np.arange(-5, 110, 10), fontsize=8)


# 4
# 器官名称和对应的体积
organs = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
# percentages = [67634,47528,50246,10014,4995,437787,104945,37865,222623,23596,1059,1209,18240,42970,17324]
percentages_pengwin = [174676, 30104, 31343, 45011, 260912, 28804, 35067, 11550, 2372, 6621, 266266, 25870, 36467, 11783 ]
plt.subplot(223)
# plt.pie(percentages_pengwin, labels=organs, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab20.colors)
plt.pie(percentages_pengwin, labels=None, autopct=None, startangle=90, colors=plt.cm.tab20.colors)
plt.axis('equal')  # 确保饼图是圆形的
# 显示图形
plt.show()

plt.savefig('imgs/pie1.png', dpi=300, bbox_inches='tight')
plt.savefig('imgs/pie1.pdf', dpi=300, bbox_inches='tight')
# print(sum(percentages_amos))

