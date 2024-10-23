import mmengine
import numpy as np
import scienceplots
import matplotlib.pyplot as plt
from scipy.stats import norm, kendalltau, spearmanr
plt.style.use(['science', 'ieee'])

def plot_scatter_with_lines(data, labels, x_labels, save_path=None):
    plt.rcParams.update({'font.size': 22})
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))

    for i, points in enumerate(data):
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        fig_i = i // int(len(data) / 2)
        inner_i = i % int(len(data) / 2)
        if fig_i == 0:
            axs[0].plot(x_values, y_values, label=labels[inner_i], marker='o')
            axs[0].set_xlabel(x_labels[fig_i])
            axs[0].set_ylabel('ImageNet Top1 Accuracy')
            # axs[0].set_title('Scatter Plot with Lines (BitOps)')
        elif fig_i == 1:
            axs[1].plot(x_values, y_values, label=labels[inner_i], marker='o')
            axs[1].set_xlabel(x_labels[fig_i])
            # axs[1].set_ylabel('ImageNet Top1 Accuracy')
            # axs[1].set_title('Scatter Plot with Lines (Params)')

        # # 添加注释
        # for j, point in enumerate(points):
        #     axs[i].annotate(f'{labels[i]}_{j+1}', (point[0], point[1]), textcoords="offset points", xytext=(0,5), ha='center')

    axs[0].legend()
    axs[1].legend()
    # axs[0].grid(True)
    # axs[1].grid(True)
    plt.subplots_adjust(hspace=0.2)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.clf()

def plot_ofa_data():
    # 4g,   #w4a4,   #w5a5,   #w6a6
    data_points = [
        # BitOps
        [(3.5, 47.66), (4, 53.09), (4.5, 57.02), (5.394, 65.67), (7.991, 67.18), (11.167, 67.21)],
        [(3.5, 63.23), (4, 65.35), (4.5, 66.57), (5.394, 67.36), (7.991, 68.66), (11.167, 68.88)],
        [(5.394, 62.01), (7.991, 67.65), (11.167, 67.69)],
        [(5.394, 48.72), (7.991, 60.29), (11.167, 63.99)],
        # Params
        [(16.822, 54.31), (18, 62.4), (19.011, 63.1), (20, 66.47), (21.199, 67.18), (23.388, 67.21)],
        [(16.822, 65.7), (18, 67.78), (19.011, 68.29), (20, 68.52), (21.199, 68.66), (23.388, 68.88)],
        [(16.822, 50), (19.011, 60), (21.199, 65), (23.388, 70)],
        [(19.011, 48.72), (21.199, 60.29), (23.388, 67.69)]
    ]

    group_labels = ['Cobits_S1', 'Cobits_S2', 'LSQ_S1', 'LSQ_S2']

    # with plt.style.context(['science', 'ieee']):
    plot_scatter_with_lines(data_points, group_labels, x_labels=['BitOps(G)', 'Params(M)'], save_path='ofa_results.png')

# plot_ofa_data()

def plot_curve_with_fill(split=2, save_path=None):
    # 生成正态分布的数据
    assert split in [2, 4], f'Unsupported split: {split}'
    mu = 0
    sigma = 0.2 if split == 2 else 0.2
    x = np.linspace(mu - split*sigma, mu + split*sigma, 100)
    y = norm.pdf(x, mu, sigma)

    # 绘制正态分布曲线
    plt.plot(x, y, '#B0A2C3', linewidth=2)

    # 标注均值和标准差
    # plt.axvline(mu, color='red', linestyle='dashed', linewidth=2, label='Mean')
    # plt.axvline(mu + sigma, color='green', linestyle='dashed', linewidth=2, label='Mean + Std Dev')
    # plt.axvline(mu - sigma, color='green', linestyle='dashed', linewidth=2, label='Mean - Std Dev')

    if split == 2:
        fill_x = np.linspace(mu - 2*sigma, mu, 100)
        fill_y = norm.pdf(fill_x, mu, sigma)
        plt.fill_between(fill_x, fill_y, color='#00C9A7', alpha=1.0)
        fill_x = np.linspace(mu, mu + 2*sigma, 100)
        fill_y = norm.pdf(fill_x, mu, sigma)
        plt.fill_between(fill_x, fill_y, color='#CC6222', alpha=1.0)
    else:        
        fill_x = np.linspace(mu - 4*sigma, mu - 2*sigma, 100)
        fill_y = norm.pdf(fill_x, mu, sigma)
        plt.fill_between(fill_x, fill_y, color='#80E8FD', alpha=1.0)
        fill_x = np.linspace(mu - 2*sigma, mu, 100)
        fill_y = norm.pdf(fill_x, mu, sigma)
        plt.fill_between(fill_x, fill_y, color='#80E4D3', alpha=1.0) 
        fill_x = np.linspace(mu, mu + 2*sigma, 100)
        fill_y = norm.pdf(fill_x, mu, sigma)
        plt.fill_between(fill_x, fill_y, color='#E1A49A', alpha=1.0)               
        fill_x = np.linspace(mu + 2*sigma, mu + 4*sigma, 100)
        fill_y = norm.pdf(fill_x, mu, sigma)
        plt.fill_between(fill_x, fill_y, color='#F5EB88', alpha=1.0)
    # 填充曲线下不同的面积部分，并标注面积
    # for i in range(split):
    #     fill_x = np.linspace(mu - i*sigma, mu + i*sigma, 100)
    #     fill_y = norm.pdf(fill_x, mu, sigma)
    #     plt.fill_between(fill_x, fill_y, color='yellow', alpha=0.3, label='68% Area')

    #     fill_x_2std = np.linspace(mu - 2*sigma, mu + 2*sigma, 100)
    #     fill_y_2std = norm.pdf(fill_x_2std, mu, sigma)
    #     plt.fill_between(fill_x_2std, fill_y_2std, color='orange', alpha=0.3, label='95% Area')

    #     fill_x_3std = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    #     fill_y_3std = norm.pdf(fill_x_3std, mu, sigma)
    #     plt.fill_between(fill_x_3std, fill_y_3std, color='red', alpha=0.3, label='99.7% Area')

    plt.xlim(-1.0, 1.0)
    plt.xticks([-1.0, -0.5, 0, 0.5, 1.0], size=15)
    plt.yticks([])
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['top'].set_color('none')
    # plt.legend()

    # 显示图形
    plt.show()

    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()

# plot_curve_with_fill(split=2, save_path='demo_split2.png')
# plot_curve_with_fill(split=4, save_path='demo_split4.png')


def plot_bits_sensitive(bits_file, layer_index_map, sensitive_file=None,
                        speedup_file=None, save_path=None):  
    bits = mmengine.load(bits_file)
    w_bits, a_bits = [], []
    for w_name, a_name in layer_index_map:
        w_bits.append(bits[w_name]['chosen'] if w_name in bits else 0)
        a_bits.append(bits[a_name]['chosen'])
    # w_bits = w_bits[1:-2]
    # a_bits = a_bits[1:-2]
    assert len(w_bits) == len(a_bits)
    x = range(len(w_bits))
    labels = range(len(w_bits))

    fig, ax1 = plt.subplots(figsize=(14, 4))
    # # 绘制柱状图
    ax1.bar(x, w_bits, tick_label=labels, color='skyblue', label='Weight')  # weight
    ax1.bar(x, -np.array(a_bits), color='salmon', label='Activation') # activation
    # ax1.get_yaxis().set_visible(False)
    ax1.tick_params(axis='y', which='both', labelleft=False)
    ax1.set_xlabel('Layer Number (Backend S0)')
    # ax1.set_ylabel('Bit-widths')
    # ax1.set_ylabel('Sensitivity')
    # # bit_ranges = range(-8, 9)
    # # yticks = [str(abs(val)) for val in bit_ranges]
    # # ax1.set_yticks(bit_ranges, yticks)
    ax1.legend(loc='center left')

    if sensitive_file is not None:
        sensitive = mmengine.load(sensitive_file)
        delta_sensitive = max(sensitive.values()) - min(sensitive.values())
        w_sensitive, a_sensitive = [], []
        for w_name, a_name in layer_index_map:        
            w_sensitive.append((sensitive[w_name] - min(sensitive.values())) / delta_sensitive if w_name in sensitive else 0)
            a_sensitive.append((sensitive[a_name] - min(sensitive.values())) / delta_sensitive)
        # w_sensitive = w_sensitive[1:-2]
        # a_sensitive = a_sensitive[1:-2]
        assert len(w_sensitive) == len(a_sensitive)
        # 创建第二个 y 轴来绘制折线图
        ax2 = ax1.twinx()
        ax2.plot(x, w_sensitive, color='green', marker='o', alpha=0.7, label='Weight Sensitivity')
        ax2.plot(x, -np.array(a_sensitive), color='blue', marker='o', alpha=0.7, label='Activatoin Sensitivity')
        # ax2.set_ylabel('Sensitivity')  # , color='green')
        # ax2.tick_params('y', colors='green')
        ax2.legend(loc='lower center')

    if speedup_file is not None:
        speedup = mmengine.load(speedup_file)
        total_speedup = sum(speedup.values())
        w_speedup, a_speedup = [], []
        for w_name, a_name in layer_index_map:        
            w_speedup.append(speedup[w_name] / total_speedup if w_name in speedup else 0)
            a_speedup.append(speedup[a_name] / total_speedup if a_name in speedup else 0)
        # w_speedup = w_speedup[1:-2]
        # a_speedup = a_speedup[1:-2]
        assert len(w_speedup) == len(a_speedup)
        # 创建第二个 y 轴来绘制折线图
        ax3 = ax2.twinx()
        ax3.plot(x, w_speedup, color='yellow', marker='o', alpha=0.7, label='BitOps Speed-UP')
        ax3.plot(x, -np.array(a_speedup), color='yellow', marker='o', alpha=0.7)
        ax3.set_ylabel('BitOps Speed-UP')
        # yticks = [str(abs(val)) for val in bit_ranges]
        # ax1.set_yticks(bit_ranges, yticks)
        ax3.legend(loc='upper center')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.clf()

# 绘制柱状图
ResNet18_S0_LAYER_INDEX_MAP = [
    ['backbone.conv1.weight_fake_quant.quant_bits', 'act_quant_bits_0'],
    ['backbone.layer1.0.conv1.weight_fake_quant.quant_bits', 'act_quant_bits_1'],
    ['backbone.layer1.0.conv2.weight_fake_quant.quant_bits', 'act_quant_bits_2'],
    ['backbone.layer1.1.conv1.weight_fake_quant.quant_bits', 'act_quant_bits_3'],
    ['backbone.layer1.1.conv2.weight_fake_quant.quant_bits', 'act_quant_bits_4'],
    ['backbone.layer2.0.conv1.weight_fake_quant.quant_bits', 'act_quant_bits_5'],
    ['backbone.layer2.0.conv2.weight_fake_quant.quant_bits', 'act_quant_bits_6'],
    ['backbone.layer2.0.downsample.0.weight_fake_quant.quant_bits', 'act_quant_bits_7'],
    ['backbone.layer2.1.conv1.weight_fake_quant.quant_bits', 'act_quant_bits_8'],
    ['backbone.layer2.1.conv2.weight_fake_quant.quant_bits', 'act_quant_bits_9'],
    ['backbone.layer3.0.conv1.weight_fake_quant.quant_bits', 'act_quant_bits_10'],
    ['backbone.layer3.0.conv2.weight_fake_quant.quant_bits', 'act_quant_bits_11'],
    ['backbone.layer3.0.downsample.0.weight_fake_quant.quant_bits', 'act_quant_bits_12'],
    ['backbone.layer3.1.conv1.weight_fake_quant.quant_bits', 'act_quant_bits_13'],
    ['backbone.layer3.1.conv2.weight_fake_quant.quant_bits', 'act_quant_bits_14'],
    ['backbone.layer4.0.conv1.weight_fake_quant.quant_bits', 'act_quant_bits_15'],
    ['backbone.layer4.0.conv2.weight_fake_quant.quant_bits', 'act_quant_bits_16'],
    ['backbone.layer4.0.downsample.0.weight_fake_quant.quant_bits', 'act_quant_bits_17'],
    ['backbone.layer4.1.conv1.weight_fake_quant.quant_bits', 'act_quant_bits_18'],
    ['backbone.layer4.1.conv2.weight_fake_quant.quant_bits', 'act_quant_bits_19'],
    ['head.fc.weight_fake_quant.quant_bits', 'act_quant_bits_20'],
]
ResNet18_S1_LAYER_INDEX_MAP = [
    ['backbone.conv1.weight_fake_quant.quant_bits', 'act_quant_bits_0'],
    ['backbone.layer1.0.conv1.weight_fake_quant.quant_bits', 'act_quant_bits_1'],
    ['backbone.layer1.0.conv2.weight_fake_quant.quant_bits', 'act_quant_bits_2'],
    ['add', 'act_quant_bits_3'],
    ['backbone.layer1.1.conv1.weight_fake_quant.quant_bits', 'act_quant_bits_4'],
    ['backbone.layer1.1.conv2.weight_fake_quant.quant_bits', 'act_quant_bits_5'],
    ['add_1',  'act_quant_bits_6'],
    ['backbone.layer2.0.conv1.weight_fake_quant.quant_bits', 'act_quant_bits_7'],
    ['backbone.layer2.0.conv2.weight_fake_quant.quant_bits', 'act_quant_bits_8'],
    ['backbone.layer2.0.downsample.0.weight_fake_quant.quant_bits', 'act_quant_bits_7'],
    ['add_2',  'act_quant_bits_9'],
    ['add_2',  'act_quant_bits_10'],
    ['backbone.layer2.1.conv1.weight_fake_quant.quant_bits', 'act_quant_bits_11'],
    ['backbone.layer2.1.conv2.weight_fake_quant.quant_bits', 'act_quant_bits_12'],
    ['add_3',  'act_quant_bits_13'],
    ['backbone.layer3.0.conv1.weight_fake_quant.quant_bits', 'act_quant_bits_14'],
    ['backbone.layer3.0.conv2.weight_fake_quant.quant_bits', 'act_quant_bits_15'],
    ['backbone.layer3.0.downsample.0.weight_fake_quant.quant_bits', 'act_quant_bits_14'],
    ['add_4',  'act_quant_bits_16'],
    ['add_4',  'act_quant_bits_17'],
    ['backbone.layer3.1.conv1.weight_fake_quant.quant_bits', 'act_quant_bits_18'],
    ['backbone.layer3.1.conv2.weight_fake_quant.quant_bits', 'act_quant_bits_19'],
    ['add_5',  'act_quant_bits_20'],
    ['backbone.layer4.0.conv1.weight_fake_quant.quant_bits', 'act_quant_bits_21'],
    ['backbone.layer4.0.conv2.weight_fake_quant.quant_bits', 'act_quant_bits_22'],
    ['backbone.layer4.0.downsample.0.weight_fake_quant.quant_bits', 'act_quant_bits_21'],
    ['add_6',  'act_quant_bits_23'],
    ['add_6',  'act_quant_bits_24'],
    ['backbone.layer4.1.conv1.weight_fake_quant.quant_bits', 'act_quant_bits_25'],
    ['backbone.layer4.1.conv2.weight_fake_quant.quant_bits', 'act_quant_bits_26'],
    ['add_7',  'act_quant_bits_27'],
    ['neck_gap',  'act_quant_bits_28'],    
    ['head.fc.weight_fake_quant.quant_bits', 'act_quant_bits_29'],
]
MBV2_S0_LAYER_INDEX_MAP = [
    ['backbone.conv1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_0'],
    ['backbone.layer1.0.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_1'],
    ['backbone.layer1.0.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_2'],
    ['backbone.layer2.0.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_3'],
    ['backbone.layer2.0.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_4'],
    ['backbone.layer2.0.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_5'],
    ['backbone.layer2.1.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_6'],
    ['backbone.layer2.1.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_7'],
    ['backbone.layer2.1.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_8'],
    ['backbone.layer3.0.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_9'],
    ['backbone.layer3.0.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_10'],
    ['backbone.layer3.0.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_11'],
    ['backbone.layer3.1.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_12'],
    ['backbone.layer3.1.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_13'],
    ['backbone.layer3.1.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_14'],
    ['backbone.layer3.2.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_15'],
    ['backbone.layer3.2.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_16'],
    ['backbone.layer3.2.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_17'],
    ['backbone.layer4.0.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_18'],
    ['backbone.layer4.0.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_19'],
    ['backbone.layer4.0.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_20'],
    ['backbone.layer4.1.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_21'],
    ['backbone.layer4.1.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_22'],
    ['backbone.layer4.1.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_23'],
    ['backbone.layer4.2.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_24'],
    ['backbone.layer4.2.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_25'],
    ['backbone.layer4.2.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_26'],
    ['backbone.layer4.3.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_27'],
    ['backbone.layer4.3.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_28'],
    ['backbone.layer4.3.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_29'],
    ['backbone.layer5.0.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_30'],
    ['backbone.layer5.0.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_31'],
    ['backbone.layer5.0.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_32'],
    ['backbone.layer5.1.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_33'],
    ['backbone.layer5.1.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_34'],
    ['backbone.layer5.1.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_35'],
    ['backbone.layer5.2.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_36'],
    ['backbone.layer5.2.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_37'],
    ['backbone.layer5.2.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_38'],
    ['backbone.layer6.0.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_39'],
    ['backbone.layer6.0.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_40'],
    ['backbone.layer6.0.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_41'],
    ['backbone.layer6.1.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_42'],
    ['backbone.layer6.1.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_43'],
    ['backbone.layer6.1.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_44'],
    ['backbone.layer6.2.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_45'],
    ['backbone.layer6.2.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_46'],
    ['backbone.layer6.2.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_47'],
    ['backbone.layer7.0.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_48'],
    ['backbone.layer7.0.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_49'],
    ['backbone.layer7.0.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_50'],
    ['backbone.conv2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_51'],
    ['head.fc.weight_fake_quant.quant_bits', 'act_quant_bits_52'],
]

MBV2_S1_LAYER_INDEX_MAP = [
    ['backbone.conv1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_0'],
    ['backbone.layer1.0.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_1'],
    ['backbone.layer1.0.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_2'],
    ['backbone.layer2.0.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_3'],
    ['backbone.layer2.0.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_4'],
    ['backbone.layer2.0.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_5'],
    ['backbone.layer2.1.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_6'],
    ['backbone.layer2.1.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_7'],
    ['backbone.layer2.1.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_8'],
    ['add','act_quant_bits_9'],
    ['backbone.layer3.0.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_10'],
    ['backbone.layer3.0.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_11'],
    ['backbone.layer3.0.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_12'],
    ['backbone.layer3.1.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_13'],
    ['backbone.layer3.1.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_14'],
    ['backbone.layer3.1.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_15'],
    ['add_1','act_quant_bits_16'],
    ['backbone.layer3.2.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_17'],
    ['backbone.layer3.2.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_18'],
    ['backbone.layer3.2.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_19'],
    ['add_2','act_quant_bits_20'],
    ['backbone.layer4.0.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_21'],
    ['backbone.layer4.0.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_22'],
    ['backbone.layer4.0.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_23'],
    ['backbone.layer4.1.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_24'],
    ['backbone.layer4.1.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_25'],
    ['backbone.layer4.1.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_26'],
    ['add_3','act_quant_bits_27'],
    ['backbone.layer4.2.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_28'],
    ['backbone.layer4.2.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_29'],
    ['backbone.layer4.2.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_30'],
    ['add_4','act_quant_bits_31'],
    ['backbone.layer4.3.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_32'],
    ['backbone.layer4.3.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_33'],
    ['backbone.layer4.3.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_34'],
    ['add_5','act_quant_bits_35'],
    ['backbone.layer5.0.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_36'],
    ['backbone.layer5.0.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_37'],
    ['backbone.layer5.0.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_38'],
    ['backbone.layer5.1.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_39'],
    ['backbone.layer5.1.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_40'],
    ['backbone.layer5.1.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_41'],
    ['add_6','act_quant_bits_42'],
    ['backbone.layer5.2.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_43'],
    ['backbone.layer5.2.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_44'],
    ['backbone.layer5.2.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_45'],
    ['add_7','act_quant_bits_46'],
    ['backbone.layer6.0.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_47'],
    ['backbone.layer6.0.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_48'],
    ['backbone.layer6.0.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_49'],
    ['backbone.layer6.1.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_50'],
    ['backbone.layer6.1.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_51'],
    ['backbone.layer6.1.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_52'],
    ['add_8','act_quant_bits_53'],
    ['backbone.layer6.2.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_54'],
    ['backbone.layer6.2.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_55'],
    ['backbone.layer6.2.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_56'],
    ['add_9','act_quant_bits_57'],
    ['backbone.layer7.0.conv.0.conv.weight_fake_quant.quant_bits', 'act_quant_bits_58'],
    ['backbone.layer7.0.conv.1.conv.weight_fake_quant.quant_bits', 'act_quant_bits_59'],
    ['backbone.layer7.0.conv.2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_60'],
    ['backbone.conv2.conv.weight_fake_quant.quant_bits', 'act_quant_bits_61'],
    ['neck_gap', 'act_quant_bits_62'],
    ['head.fc.weight_fake_quant.quant_bits', 'act_quant_bits_63'],
]

ResNet18_S0_SPEEDUP_FILE='/home/wangshiguang/mnodes/NAS-MQBench/mmrazor/work_dirs/cobits_weightonly_resnet18_supernet_8xb64_in1k/20240124_132736/bitops_speedup.json'
ResNet18_S0_Sensitive_FILE='/home/wangshiguang/mnodes/NAS-MQBench/mmrazor/work_dirs/cobits_weightonly_resnet18_supernet_8xb64_in1k/20240124_132736/indicators.json'
ResNet18_S1_SPEEDUP_FILE='/home/wangshiguang/mnodes/NAS-MQBench/mmrazor/work_dirs/cobits_snpe_resnet18_supernet_8xb64_in1k/20240125_061757/bitops_speedup.json'
ResNet18_S1_Sensitive_FILE='/home/wangshiguang/mnodes/NAS-MQBench/mmrazor/work_dirs/cobits_snpe_resnet18_supernet_8xb64_in1k/20240125_061757/indicators.json'
MobileNetV2_S0_SPEEDUP_FILE='/home/wangshiguang/mnodes/NAS-MQBench/mmrazor/work_dirs/cobits_weightonly_mbv2_supernet_8xb64_in1k/20240129_004308/bitops_speedup.json'
MobileNetV2_S0_Sensitive_FILE='/home/wangshiguang/mnodes/NAS-MQBench/mmrazor/work_dirs/cobits_weightonly_mbv2_supernet_8xb64_in1k/20240129_004308/indicators.json'
MobileNetV2_S1_SPEEDUP_FILE='/home/wangshiguang/mnodes/NAS-MQBench/mmrazor/work_dirs/cobits_snpe_mbv2_supernet_8xb64_in1k/20240218_050835/bitops_speedup.json'
MobileNetV2_S1_Sensitive_FILE='/home/wangshiguang/mnodes/NAS-MQBench/mmrazor/work_dirs/cobits_snpe_mbv2_supernet_8xb64_in1k/20240218_050835/indicators.json'
# plot_bits_sensitive('/home/wangshiguang/mnodes/NAS-MQBench/mmrazor/work_dirs/cobits_weightonly_resnet18_search_8xb64_in1k/20240202_030948/best_fix_subnet.yaml',
#                     layer_index_map=ResNet18_S0_LAYER_INDEX_MAP,
#                     sensitive_file=ResNet18_S0_Sensitive_FILE,
#                     speedup_file=ResNet18_S0_SPEEDUP_FILE,
#                     save_path='ResNet18-3bit-S0.png')
plot_bits_sensitive('/home/wangshiguang/mnodes/NAS-MQBench/mmrazor/work_dirs/cobits_weightonly_resnet18_search_8xb64_in1k/20240204_070053/best_fix_subnet.yaml',
                    layer_index_map=ResNet18_S0_LAYER_INDEX_MAP,
                    sensitive_file=ResNet18_S0_Sensitive_FILE,
                    speedup_file=ResNet18_S0_SPEEDUP_FILE,
                    save_path='ResNet18-4bit-S0.png')
# plot_bits_sensitive('/home/wangshiguang/mnodes/NAS-MQBench/mmrazor/work_dirs/cobits_snpe_resnet18_search_8xb64_in1k/20240207_071927/best_fix_subnet.yaml',
#                     layer_index_map=ResNet18_S1_LAYER_INDEX_MAP,
#                     sensitive_file=ResNet18_S1_Sensitive_FILE,
#                     speedup_file=ResNet18_SPEEDUP_FILE,   
#                     save_path='ResNet18-3bit-S1.png')
plot_bits_sensitive('/home/wangshiguang/mnodes/NAS-MQBench/mmrazor/work_dirs/cobits_snpe_resnet18_search_8xb64_in1k/20240125_093155/best_fix_subnet.yaml',
                    layer_index_map=ResNet18_S1_LAYER_INDEX_MAP,
                    sensitive_file=ResNet18_S1_Sensitive_FILE,
                    speedup_file=ResNet18_S1_SPEEDUP_FILE,
                    save_path='ResNet18-4bit-S1.png')


# plot_bits_sensitive('/home/wangshiguang/mnodes/NAS-MQBench/mmrazor/work_dirs/cobits_weightonly_mbv2_search_8xb64_in1k/20240207_142833/best_fix_subnet.yaml',
#                     layer_index_map=MBV2_S0_LAYER_INDEX_MAP,
#                     save_path='MobileNetV2-3bit-S0.png')
plot_bits_sensitive('/home/wangshiguang/mnodes/NAS-MQBench/mmrazor/work_dirs/cobits_weightonly_mbv2_search_8xb64_in1k/20240206_132302/best_fix_subnet.yaml',
                    layer_index_map=MBV2_S0_LAYER_INDEX_MAP,
                    sensitive_file=MobileNetV2_S0_Sensitive_FILE,
                    speedup_file=MobileNetV2_S0_SPEEDUP_FILE,                    
                    save_path='MobileNetV2-4bit-S0.png')

plot_bits_sensitive('/home/wangshiguang/mnodes/NAS-MQBench/mmrazor/work_dirs/cobits_snpe_mbv2_search_8xb64_in1k/20240218_084405/best_fix_subnet.yaml',
                    layer_index_map=MBV2_S1_LAYER_INDEX_MAP,
                    sensitive_file=MobileNetV2_S1_Sensitive_FILE,
                    speedup_file=MobileNetV2_S1_SPEEDUP_FILE,                    
                    save_path='MobileNetV2-4bit-S1.png')


def calculate_rank_correlation(matrix_files, save_path=None):
    def get_element_position(sequence, element):
        sorted_seq = np.sort(sequence)
        element_position = np.where(sorted_seq == element)[0][0]
        return element_position

    matrix = []
    for k, mf in matrix_files.items():
        v = mmengine.load(mf)
        sequence = np.array(list(v.values()))
        # import pdb; pdb.set_trace()
        sequence = [get_element_position(sequence, vv) for vv in sequence]
        matrix.append(sequence)
    matrix = np.array(matrix)
    rows, _ = matrix.shape

    def get_plot_corrections(ax, mode='Spearman'):
        assert mode in ['Spearman', 'Kendall']
        correlations = np.ones((rows, rows))
        for i in range(rows):
            for j in range(i + 1, rows):
                if mode == 'Spearman':
                    corr, _ = spearmanr(matrix[i], matrix[j])
                else:
                    corr, _ = kendalltau(matrix[i], matrix[j])
                correlations[i, j] = correlations[j, i] = corr

        im = ax.imshow(correlations, cmap='viridis', interpolation='nearest')
        for i in range(correlations.shape[0]):
            for j in range(correlations.shape[1]):
                text = ax.text(j, i, f'{correlations[i, j]:.2f}',
                               ha='center', va='center', color='w')
        plt.colorbar(im, label=f'{mode} Rank Correlation')
        ax.set_xticks([])
        ax.set_yticks([])
        print(f'{mode}: {correlations}')

    fig, ax = plt.subplots(2, 1)
    print(f'{save_path}')
    # title = plt.title('Rank Correlation Matrix')
    # title.set_position([0.1, -0.5])    
    get_plot_corrections(ax[0], mode='Spearman')
    get_plot_corrections(ax[1], mode='Kendall')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.clf()


ResNet18_S0_matrix_files = {
    'EP3': '/home/wangshiguang/mnodes/NAS-MQBench/mmrazor/work_dirs/cobits_weightonly_resnet18_supernet_8xb64_in1k/ep3_20240221_135028/indicators.json',
    'EP5': '/home/wangshiguang/mnodes/NAS-MQBench/mmrazor/work_dirs/cobits_weightonly_resnet18_supernet_8xb64_in1k/20240124_132736/indicators.json',
    'EP10': '/home/wangshiguang/mnodes/NAS-MQBench/mmrazor/work_dirs/cobits_weightonly_resnet18_supernet_8xb64_in1k/ep10_20240221_154431/indicators.json'
}
calculate_rank_correlation(ResNet18_S0_matrix_files, save_path='ResNet18-Matrix-S0.png')

ResNet18_S1_matrix_files = {
    'ResNet18-S1-EP3': '/home/wangshiguang/mnodes/NAS-MQBench/mmrazor/work_dirs/cobits_snpe_resnet18_supernet_8xb64_in1k/ep3_20240222_005805/indicators.json',
    'ResNet18-S1-EP5': '/home/wangshiguang/mnodes/NAS-MQBench/mmrazor/work_dirs/cobits_snpe_resnet18_supernet_8xb64_in1k/20240125_061757/indicators.json',
    'ResNet18-S1-EP10': '/home/wangshiguang/mnodes/NAS-MQBench/mmrazor/work_dirs/cobits_snpe_resnet18_supernet_8xb64_in1k/ep10_20240221_155358/indicators.json'
}
calculate_rank_correlation(ResNet18_S1_matrix_files, save_path='ResNet18-Matrix-S1.png')


ResNet18_S2_matrix_files = {
    'ResNet18-S2-EP3': '/home/wangshiguang/mnodes/NAS-MQBench/mmrazor/work_dirs/cobits_openvinox_resnet18_supernet_8xb64_in1k/ep3_20240221_154920/indicators.json',
    'ResNet18-S2-EP5': '/home/wangshiguang/mnodes/NAS-MQBench/mmrazor/work_dirs/cobits_openvinox_resnet18_supernet_8xb64_in1k/20240204_085242/indicators.json',
    'ResNet18-S2-EP10': '/home/wangshiguang/mnodes/NAS-MQBench/mmrazor/work_dirs/cobits_openvinox_resnet18_supernet_8xb64_in1k/ep10_20240222_001729/indicators.json'
}
calculate_rank_correlation(ResNet18_S2_matrix_files, save_path='ResNet18-Matrix-S2.png')

