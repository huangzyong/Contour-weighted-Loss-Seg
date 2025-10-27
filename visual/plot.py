import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(1, 14, 14)
num_cls = [100, 45, 7, 1, 100, 66, 36, 11, 1, 1, 100, 64, 34, 9]
plt.figure(figsize=(10,8))
x = np.linspace(1, 4, 4)
num_cls_1 = [100, 45, 7, 1]
plt.plot(x, num_cls_1, c='#ff7f0e', ls='-', linewidth=1.2, marker='*', ms=8, alpha=0.5, label='sacrum fragment')
x = np.linspace(5, 10, 6)
num_cls_1 = [100, 66, 36, 11, 1, 1]
plt.plot(x, num_cls_1, c='#2ca02c', ls='-', linewidth=1.2, marker='d', ms=8, alpha=0.5, label='left hipbone fragment')
x = np.linspace(11, 14, 4)
num_cls_1 = [100, 64, 34, 9]
plt.plot(x, num_cls_1, c='#d62728', ls='-', linewidth=1.2, marker='p', ms=8, alpha=0.5, label='right hipbone fragment')
plt.rcParams.update({'font.size': 16}) 

plt.legend(loc='upper right', bbox_to_anchor=(1.0, 0.9))
plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.5)
plt.xticks(np.arange(0, 16, 1), fontsize=14)
plt.yticks(np.arange(-5, 110, 10), fontsize=14)

# plt.ylabel("Number of Instances", fontsize=14)
# plt.xlabel("Category", fontsize=14)
plt.show()

plt.savefig('imgs/pengwwin_category.png', dpi=300, bbox_inches='tight')
plt.savefig('imgs/pengwwin_category.pdf', dpi=300, bbox_inches='tight')