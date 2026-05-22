import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import matplotlib.patches as mpatches

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# 颜色和hatch模式设置
colors = [
    '#AF7AC5',  # 淡紫（对应 K=32）
    '#48C9B0',  # 清爽青绿
    '#DC7633',  # 温暖棕橙（与 Jetson Xavier 线相近）
    '#EC7063' ,  # 柔和粉红（类似圈中的红点）
    '#FFA726', # 柔和橙黄（与上图中 K=16 线类似）
    '#4A86E8' # 柔和天蓝（来自相似度图）

]

hatches = [ '--', '\\\\', 'xx', '..', '||', '//']
edgecolor = 'white'

# 模型标签
labels = ["BOLT (Cloud-Only)","AKS (Cloud-Only)","BOLT (Edge-Cloud)","AKS (Edge-Cloud)","Vanilla","Vrag"]

# ================= 改为2行2列布局 =================
fig, axs = plt.subplots(2, 2, figsize=(12, 5.1))  # 调整为2x2布局
fig.subplots_adjust(wspace=0.18, hspace=0.63)  # 调整水平和垂直间距

# 100mbps数据重新排列
edge_100 = [0.96, 0.99, 586.458584, 626.458584, 576.458584, 1.440620]
comm_100 = [49.184, 46.184, 1.358576, 1.038576, 1.148576, 0.5570560]
cloud_100 = [24.49069314, 26.490643174, 3.84480355, 4.20204289, 3.24283966, 2.6980579]
total_100 = [74.63469314,73.66464317, 613.2005476, 631.1577869, 580.0985837,4.8957339]

def plot_subfig(ax, time, title):
    # 使用对数坐标解决大跨度问题
    ax.set_yscale('log')
    
    # 堆叠柱状图设置
    bar_width = 0.7
    x = np.arange(len(labels))
    
    # 绘制柱状图
    for i in range(6):
        ax.bar(x[i], time[i], bar_width, 
               color=colors[i], 
               edgecolor='white', 
               hatch=hatches[i], 
               label=labels[i])
    
    # 添加网格线
    for y in [1, 3.16, 10, 31.62, 100, 316.23, 1000]:
        ax.axhline(y=y, linestyle='--', color='lightgray', linewidth=0.8, zorder=0)
    
    ax.set_title(title, fontsize=21, pad=12)
    ax.set_xticks([])
    ax.set_xlabel("Baselines", fontsize=20, labelpad=10)  # 添加统一的 x 轴 label
    ax.tick_params(axis='y', which='both', labelsize=15)

# ================= 调整布局 =================
# 第一个子图 - 位置(0,0)
plot_subfig(axs[0, 0], cloud_100, "Cloud Process Latency")
axs[0, 0].set_ylabel('Time (s)', fontsize=21)
axs[0, 0].set_ylim(0.1, 120)


# 第二个子图 - 位置(0,1)
plot_subfig(axs[0, 1], comm_100, "Communication Latency")
axs[0, 1].set_ylim(0.1, 120)

# 第三个子图 - 位置(1,0)
plot_subfig(axs[1, 0], edge_100, "Edge Process Latency")
axs[1, 0].set_ylabel('Time (s)', fontsize=21)
axs[1, 0].set_ylim(0.1, 2000)
# axs[1,0].set_xlabel("Baselines", fontsize=23, labelpad=10)  # 添加统一的 x 轴 label

# 第四个子图 - 位置(1,1) 留空或添加说明
plot_subfig(axs[1, 1], total_100, "Total Latency")
axs[1, 1].set_ylim(0.1, 2000)
# axs[1,1].set_xlabel("Baselines", fontsize=23, labelpad=10)  # 添加统一的 x 轴 label


# ================= 创建统一图例 =================
handles = [
    mpatches.Patch(facecolor=colors[i], hatch=hatches[i], edgecolor=edgecolor, label=labels[i])
    for i in range(6)
]

fig.legend(handles=handles, 
           loc='upper center', 
           ncol=3,
           fontsize=16.5,
           # 增大宽度的参数
           handletextpad=1.3,      # 增加100%（原1.0→2.0）句柄与文本的间距
           columnspacing=2.5,      # 增加66%（原1.5→2.5）列间间距
           borderpad=0.8,          # 增加50%（原0.8→1.2）图例框内边距
           handlelength=2.0,       # 增加50%（原2.0→3.0）色块长度
           # 减小高度的参数
           handleheight=0.8,       # 减小38%（原0.8→0.5）色块高度
           labelspacing=0.4,       # 减少图例项间垂直间距
           # 布局控制
           bbox_to_anchor=(0.503, 1.175)  # 图例位置进一步提升到画布上方12%
)

# 调整布局为图例留出空间[5](@ref)
# plt.tight_layout(pad=0.0, rect=(0, 0, 1, 0.92))  # 调整rect参数为图例留出顶部空间


# 保存PDF
pdf = PdfPages('breakdown_latency_comp.pdf')
pdf.savefig(fig, bbox_inches='tight')
pdf.close()

# plt.tight_layout(pad=2.0, rect=(0, 0, 1, 0.9))  # 调整布局为图例留出空间
plt.show()