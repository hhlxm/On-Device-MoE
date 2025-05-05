from collections import defaultdict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import json
from pathlib import Path


def cosine_similarity_scipy(vec1, vec2):
    return 1 - cosine(vec1, vec2)

def plot_cosine_similarity_grid(
    frequencies,
    layers=range(0, 32),
    token_start=1,
    token_end=None,
    base_width_per_plot=10,  # 每个子图的基准宽度（英寸）
    base_height_per_plot=10,  # 每个子图的基准高度（英寸）
    max_cols=10,  # 最大列数
    bar_color='lightblue',
    bar_edgecolor='black',
    bar_width=0.8,
    y_ticks=np.arange(0, 1.2, 0.2),
    base_title_fontsize=22,
    base_label_fontsize=20,
    base_tick_fontsize=25,
    output_path=None,
    output_format="svg",
    dpi=600,
    bbox_inches="tight"
):
    """
    Plots a grid of cosine similarity bar charts for a specified range of tokens, with dynamic figure size.

    Parameters:
        frequencies (dict): A nested dictionary containing frequency data.
        layers (iterable): The layers (e.g., layers) to plot on the x-axis.
        token_start (int): Start index of the token range to plot (inclusive).
        token_end (int): End index of the token range to plot (exclusive). If None, defaults to the maximum available tokens.
        base_width_per_plot (float): Base width per subplot in inches.
        base_height_per_plot (float): Base height per subplot in inches.
        max_cols (int): Maximum number of columns in the subplot grid.
        bar_color (str): Color of the bars.
        bar_edgecolor (str): Edge color of the bars.
        bar_width (float): Width of the bars.
        y_ticks (array-like): Y-axis ticks for the plots.
        base_title_fontsize (int): Base font size for subplot titles.
        base_label_fontsize (int): Base font size for axis labels.
        base_tick_fontsize (int): Base font size for tick labels.
        output_path (str): Path to save the output figure.
        output_format (str): Format of the output file (e.g., "svg", "png").
        dpi (int): DPI for the saved figure.
        bbox_inches (str): Bounding box adjustment for saving the figure.
    """
    # 确定 token 范围
    max_tokens = len(frequencies[list(frequencies.keys())[0]]['routed']) - 1
    print(max_tokens)
    token_end = min(token_end or max_tokens, max_tokens)
    token_range = range(token_start, token_end)
    num_plots = len(token_range)

    if num_plots == 0:
        raise ValueError("Token range is empty. Please specify a valid token_start and token_end.")

    # 计算余弦相似度
    all_vec_res = []
    for j in token_range:
        vec_res = []
        for i in layers:
            similarity = cosine_similarity_scipy(frequencies[i]['routed'][j], frequencies[i]['routed'][j + 1])
            vec_res.append(float(similarity))
        all_vec_res.append(vec_res)

    # 动态计算 nrows 和 ncols
    ncols = min(max_cols, num_plots)  # 列数不超过 max_cols
    nrows = (num_plots + ncols - 1) // ncols  # 向上取整计算行数

    # 动态调整 figsize
    width = ncols * base_width_per_plot
    height = nrows * base_height_per_plot
    figsize = (width, height)

    # 动态调整字体大小（根据图片大小缩放）
    scale_factor = min(width / 40, height / 36)  # 基于原始默认 figsize=(40, 36) 的缩放比例
    title_fontsize = base_title_fontsize 
    label_fontsize = base_label_fontsize 
    tick_fontsize = base_tick_fontsize 

    # 创建子图
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey=True)
    if num_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # 处理单个子图的情况

    # 绘制每个 token 对的柱状图
    for idx, j in enumerate(token_range):
        ax = axes[idx]
        vec_res = all_vec_res[idx]
        ax.bar(layers, vec_res, color=bar_color, edgecolor=bar_edgecolor, width=bar_width)
        ax.set_title(f'Token{j} & Token{j+1}', fontsize=title_fontsize, pad=10)
        ax.set_xlabel("Layer#", fontsize=label_fontsize)
        ax.set_ylim(0, 1.1)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis='x', labelsize=tick_fontsize)
        ax.tick_params(axis='y', labelsize=tick_fontsize)

    # 关闭未使用的子图
    for j in range(num_plots, nrows * ncols):
        axes[j].axis('off')

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    if output_path is not None:
        plt.savefig(output_path, format=output_format, dpi=dpi, bbox_inches=bbox_inches)
    plt.show()

def plot_cosine_similarity_avg(
    frequencies,
    layers=range(0, 32),
    token_start=0,
    token_end=None,
    base_width_per_plot=10,  # 单个图的基准宽度（英寸）
    base_height_per_plot=6,  # 单个图的基准高度（英寸）
    line_color='red',
    line_style='-',
    marker='o',
    y_ticks=np.arange(0, 1.2, 0.2),
    base_title_fontsize=12,
    base_label_fontsize=10,
    base_tick_fontsize=15,
    output_path=None,
    output_format="svg",
    dpi=600,
    bbox_inches="tight"
):
    """
    Plots a line chart of the average cosine similarity across all token pairs in the specified range, per layer.

    Parameters:
        frequencies (dict): A nested dictionary containing frequency data.
        layers (iterable): The layers (e.g., layers) to plot on the x-axis.
        token_start (int): Start index of the token range to compute (inclusive).
        token_end (int): End index of the token range to compute (exclusive). If None, defaults to the maximum available tokens.
        base_width_per_plot (float): Base width of the plot in inches.
        base_height_per_plot (float): Base height of the plot in inches.
        line_color (str): Color of the line.
        line_style (str): Line style (e.g., '-', '--', '-.', ':').
        marker (str): Marker style for data points (e.g., 'o', 's', '^').
        y_ticks (array-like): Y-axis ticks for the plot.
        base_title_fontsize (int): Font size for the plot title.
        base_label_fontsize (int): Font size for axis labels.
        base_tick_fontsize (int): Font size for tick labels.
        output_path (str): Path to save the output figure and JSON file.
        output_format (str): Format of the output file (e.g., "svg", "png").
        dpi (int): DPI for the saved figure.
        bbox_inches (str): Bounding box adjustment for saving the figure.
    """
    # 确定 token 范围
    max_tokens = len(frequencies[list(frequencies.keys())[0]]['routed']) - 1
    token_end = min(token_end or max_tokens, max_tokens)
    token_range = range(token_start, token_end)
    num_pairs = len(token_range)

    if num_pairs == 0:
        raise ValueError("Token range is empty. Please specify a valid token_start and token_end.")

    # 计算每一层的余弦相似度
    layer_similarities = np.zeros(len(layers))  # 存储每一层的平均相似度
    for j in token_range:
        for idx, i in enumerate(layers):
            similarity = cosine_similarity_scipy(frequencies[i]['routed'][j], frequencies[i]['routed'][j + 1])
            layer_similarities[idx] += similarity
    
    # 计算平均值
    layer_similarities /= num_pairs

    # 设置图形大小
    figsize = (base_width_per_plot, base_height_per_plot)

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制折线图
    ax.plot(layers, layer_similarities, color=line_color, linestyle=line_style, marker=marker, label="Average Similarity")

    # 设置标题和标签
    ax.set_title(f'Average Cosine Similarity', fontsize=base_title_fontsize, pad=10)
    ax.set_xlabel("Layer#", fontsize=base_label_fontsize)
    ax.set_ylabel("Average Cosine Similarity", fontsize=base_label_fontsize)

    # 设置刻度
    ax.set_ylim(0, 1.1)
    ax.set_yticks(y_ticks)
    ax.set_xticks(list(layers))
    ax.tick_params(axis='x', labelsize=base_tick_fontsize)
    ax.tick_params(axis='y', labelsize=base_tick_fontsize)

    # 添加图例
    ax.legend(fontsize=base_label_fontsize)

    # 调整布局
    plt.tight_layout()

    # 保存图形和数据
    if output_path is not None:
        # 确保输出路径存在
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存图形
        plt.savefig(output_path, format=output_format, dpi=dpi, bbox_inches=bbox_inches)

        # 构造 JSON 文件路径
        json_path = output_path.with_suffix('.json') if isinstance(output_path, Path) else Path(output_path).with_suffix('.json')

        # 保存 layer_similarities 数据为 JSON 文件
        with open(json_path, 'w') as f:
            json.dump({
                "layers": list(layers),
                "similarities": layer_similarities.tolist()
            }, f, indent=4)

    plt.show()
    return layer_similarities

def plot_cosine_similarity_avg_cross_batch(
    frequencies,
    layers=range(0, 32),
    token_start=1,
    token_end=None,
    base_width_per_plot=10,  # 单个图的基准宽度（英寸）
    base_height_per_plot=6,  # 单个图的基准高度（英寸）
    line_color='red',
    line_style='-',
    marker='o',
    y_ticks=np.arange(0, 1.2, 0.2),
    base_title_fontsize=12,
    base_label_fontsize=10,
    base_tick_fontsize=15,
    output_path=None,
    output_format="svg",
    dpi=600,
    bbox_inches="tight"
):
    """
    Plots a line chart of the average cosine similarity across all token pairs in the specified range, per layer.

    Parameters:
        frequencies (dict): A nested dictionary containing frequency data.
        layers (iterable): The layers (e.g., layers) to plot on the x-axis.
        token_start (int): Start index of the token range to compute (inclusive).
        token_end (int): End index of the token range to compute (exclusive). If None, defaults to the maximum available tokens.
        base_width_per_plot (float): Base width of the plot in inches.
        base_height_per_plot (float): Base height of the plot in inches.
        line_color (str): Color of the line.
        line_style (str): Line style (e.g., '-', '--', '-.', ':').
        marker (str): Marker style for data points (e.g., 'o', 's', '^').
        y_ticks (array-like): Y-axis ticks for the plot.
        base_title_fontsize (int): Font size for the plot title.
        base_label_fontsize (int): Font size for axis labels.
        base_tick_fontsize (int): Font size for tick labels.
        output_path (str): Path to save the output figure and JSON file.
        output_format (str): Format of the output file (e.g., "svg", "png").
        dpi (int): DPI for the saved figure.
        bbox_inches (str): Bounding box adjustment for saving the figure.
    """
    whole_layer_similarities = np.zeros(len(layers))
    cnt=0
    for k, freq_k in frequencies.items():  # 直接解包 k 对应的字典，减少字典查询
        max_tokens = len(freq_k[list(freq_k.keys())[0]]['routed']) - 1
        # print(max_tokens)
        t_end = min(token_end or max_tokens, max_tokens)
        token_range = np.arange(token_start, t_end)  # 用 NumPy 数组加速
        
        if len(token_range) == 0:
            print("skip: Token range is empty. Please specify a valid token_start and token_end.")
            # raise ValueError("Token range is empty. Please specify a valid token_start and token_end.")
            continue

        cnt+=1
        # 计算每一层的余弦相似度
        layer_similarities = np.zeros(len(layers))

        for idx, i in enumerate(layers):  
            routed_i = freq_k[i]['routed']  # 避免重复字典查询

            # 计算所有 token 对的余弦相似度并取均值（向量化）
            similarities = np.array([
                cosine_similarity_scipy(routed_i[j], routed_i[j + 1]) for j in token_range
            ])
            layer_similarities[idx] = similarities.mean()  # 直接计算均值

        whole_layer_similarities += layer_similarities

    print(f"Total batches processed: {cnt}")
    whole_layer_similarities /= cnt

    layer_similarities = whole_layer_similarities

    # 设置图形大小
    figsize = (base_width_per_plot, base_height_per_plot)

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制折线图
    ax.plot(layers, layer_similarities, color=line_color, linestyle=line_style, marker=marker, label="Average Similarity")

    # 设置标题和标签
    ax.set_title(f'Average Cosine Similarity', fontsize=base_title_fontsize, pad=10)
    ax.set_xlabel("Layer#", fontsize=base_label_fontsize)
    ax.set_ylabel("Average Cosine Similarity", fontsize=base_label_fontsize)

    # 设置刻度
    ax.set_ylim(0, 1.1)
    ax.set_yticks(y_ticks)
    ax.set_xticks(list(layers))
    ax.tick_params(axis='x', labelsize=base_tick_fontsize)
    ax.tick_params(axis='y', labelsize=base_tick_fontsize)

    # 添加图例
    ax.legend(fontsize=base_label_fontsize)

    # 调整布局
    plt.tight_layout()

    # 保存图形和数据
    if output_path is not None:
        # 确保输出路径存在
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存图形
        plt.savefig(output_path, format=output_format, dpi=dpi, bbox_inches=bbox_inches)

        # 构造 JSON 文件路径
        json_path = output_path.with_suffix('.json') if isinstance(output_path, Path) else Path(output_path).with_suffix('.json')

        # 保存 layer_similarities 数据为 JSON 文件
        with open(json_path, 'w') as f:
            json.dump({
                "layers": list(layers),
                "similarities": layer_similarities.tolist()
            }, f, indent=4)

    plt.show()
    return layer_similarities

def plot_layer_sums(
    frequencies,
    title="Sum of Values per Layer",
    base_width=15,
    base_height=10,
    line_color='blue',
    line_style='-',
    marker='o',
    base_title_fontsize=16,
    base_label_fontsize=14,
    base_tick_fontsize=12,
    output_path=None,
    output_format="svg",
    dpi=600,
    bbox_inches="tight"
):
    """
    Draws a line chart of the sum of values per layer.

    Parameters:
        frequencies (dict): Dictionary with layer indices as keys and numpy arrays as values.
        base_width (float): Base width of the plot in inches.
        base_height (float): Base height of the plot in inches.
        line_color (str): Color of the line.
        line_style (str): Line style (e.g., '-', '--', '-.', ':').
        marker (str): Marker style for data points (e.g., 'o', 's', '^').
        base_title_fontsize (int): Font size for the plot title.
        base_label_fontsize (int): Font size for axis labels.
        base_tick_fontsize (int): Font size for tick labels.
        output_path (str or Path): Path to save the plot and data. If None, only display.
        output_format (str): Format of the output file (e.g., "svg", "png").
        dpi (int): DPI for the saved figure.
        bbox_inches (str): Bounding box adjustment for saving.
    Returns:
        layer_sums (list): List of sums for each layer.
    """
    # 计算每一层的和
    layers = sorted(frequencies.keys())
    layer_sums = [np.sum(frequencies[layer]) for layer in layers]

    # 创建折线图
    fig, ax = plt.subplots(figsize=(base_width, base_height))
    ax.plot(layers, layer_sums, color=line_color, linestyle=line_style, marker=marker, linewidth=2, markersize=8)
    # 设置标题和标签
    ax.set_title(title, fontsize=base_title_fontsize, pad=10)
    ax.set_xlabel("Layer#", fontsize=base_label_fontsize)
    ax.set_ylabel("Sum of Values", fontsize=base_label_fontsize)

    # 设置刻度和范围
    ax.set_xticks(layers)
    ax.set_ylim(0, max(layer_sums) * 1.1)
    ax.tick_params(axis='x', labelsize=base_tick_fontsize)
    ax.tick_params(axis='y', labelsize=base_tick_fontsize)
    ax.grid(True, linestyle='--', alpha=0.7)

    # 调整布局
    plt.tight_layout()

    # 保存图形和数据
    if output_path is not None:
        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存折线图
        plt.savefig(output_path, format=output_format, dpi=dpi, bbox_inches=bbox_inches)

        # 保存数据为 JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump({
                "layers": layers,
                "layer_sums": layer_sums
            }, f, indent=4)

    # 显示图形
    plt.show()
    return layer_sums


def draw_batch(
    frequencies,
    title="Average Routed Value Across Layers",
    base_width=15,
    base_height=10,
    line_color='blue',
    line_style='-',
    marker='o',
    y_ticks=np.arange(0, 1.1, 0.1),
    base_title_fontsize=16,
    base_label_fontsize=14,
    base_tick_fontsize=12,
    output_path=None,
    output_format="svg",
    dpi=600,
    bbox_inches="tight"
):
    """
    Draws a line chart of average routed values across layers, averaged over batches.

    Parameters:
        frequencies (dict): Nested dictionary with batch data, e.g., {batch_idx: {layer_idx: {'routed': float}}}
        title (str): Title of the plot.
        base_width (float): Base width of the plot in inches.
        base_height (float): Base height of the plot in inches.
        line_color (str): Color of the line.
        line_style (str): Line style (e.g., '-', '--', '-.', ':').
        marker (str): Marker style for data points (e.g., 'o', 's', '^').
        y_ticks (array-like): Y-axis ticks for the plot.
        base_title_fontsize (int): Font size for the plot title.
        base_label_fontsize (int): Font size for axis labels.
        base_tick_fontsize (int): Font size for tick labels.
        output_path (str or Path): Path to save the plot and data. If None, only display.
        output_format (str): Format of the output file (e.g., "svg", "png").
        dpi (int): DPI for the saved figure.
        bbox_inches (str): Bounding box adjustment for saving.
    Returns:
        average_routed_values (np.ndarray): Array of averaged routed values across layers.
    """
    # 动态确定批次数和层数
    batch_indices = list(frequencies.keys())
    if not batch_indices:
        raise ValueError("Frequencies dictionary is empty.")

    # 获取所有层索引
    all_layers = set()
    for batch_idx in batch_indices:
        all_layers.update(frequencies[batch_idx].keys())
    moe_layers = sorted(list(all_layers))  # 按顺序排列层索引
    num_moe_layers = len(moe_layers)
    num_batches = len(batch_indices)

    # 创建中间矩阵（批次 x 层数）用于计算平均值
    routed_matrix = np.zeros((num_batches, num_moe_layers))

    # 填充矩阵
    for batch_idx_idx, batch_idx in enumerate(batch_indices):
        for layer_idx, layer in enumerate(moe_layers):
            if layer in frequencies[batch_idx]:
                routed_matrix[batch_idx_idx, layer_idx] = frequencies[batch_idx][layer]["routed"]
            else:
                routed_matrix[batch_idx_idx, layer_idx] = 0

    # 对每个层取批次平均值，忽略全零的情况
    average_routed_values = np.zeros(num_moe_layers)
    for layer_idx in range(num_moe_layers):
        valid_values = routed_matrix[:, layer_idx][routed_matrix[:, layer_idx] != 0]
        if valid_values.size > 0:
            average_routed_values[layer_idx] = np.mean(valid_values)
        else:
            average_routed_values[layer_idx] = 0

    # 创建折线图
    plt.figure(figsize=(base_width, base_height))
    plt.plot(moe_layers, average_routed_values, 
             color=line_color, linestyle=line_style, marker=marker, 
             linewidth=2, markersize=8)

    # 设置标题和标签
    plt.title(title, fontsize=base_title_fontsize, pad=10)
    plt.xlabel("Layer Index", fontsize=base_label_fontsize)
    plt.ylabel("Average Routed Value", fontsize=base_label_fontsize)

    # 设置刻度和范围
    plt.xticks(moe_layers, rotation=45, ha="right", fontsize=base_tick_fontsize)
    plt.yticks(y_ticks, fontsize=base_tick_fontsize)
    plt.ylim(0, max(1.0, average_routed_values.max() * 1.1))  # 动态调整上限

    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)

    # 调整布局
    plt.tight_layout()

    # 保存图形和数据
    if output_path is not None:
        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存折线图
        plt.savefig(output_path, format=output_format, dpi=dpi, bbox_inches=bbox_inches)

        # 保存数据为 JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump({
                "layers": moe_layers,
                "average_routed_values": average_routed_values.tolist()
            }, f, indent=4)

    # 显示图形
    plt.show()


def draw(
    frequencies,
    title="Expert Activation Heatmap",
    base_width=15,
    base_height=10,
    cmap="OrRd",
    vmin=0,
    vmax=1,
    annot=False,
    output_path=None,
    output_format="svg",
    dpi=600,
    bbox_inches="tight"
):
    """
    Draws a heatmap of expert activation probabilities across layers and saves the activation matrix.

    Parameters:
        frequencies (dict): {layer_idx: {'routed': {expert_idx: frequency}}}
        title (str): Title of the heatmap.
        base_width (float): Width of the plot in inches.
        base_height (float): Height of the plot in inches.
        cmap (str): Colormap for the heatmap (e.g., "OrRd", "YlGnBu").
        vmin (float): Minimum value for the color scale.
        vmax (float): Maximum value for the color scale.
        annot (bool): Whether to annotate cells with values.
        output_path (str or Path): Path to save the heatmap and matrix (e.g., '/path/to/plot'). If None, only display.
        output_format (str): Format of the output file (e.g., "svg", "png").
        dpi (int): DPI for the saved figure.
        bbox_inches (str): Bounding box adjustment for saving.
    Returns:
        activation_matrix (np.ndarray): The activation probability matrix.
    """
    # 动态确定层数和专家数
    moe_layers = sorted([layer_idx for layer_idx in frequencies.keys() ])  # 排除第 0 层
    num_moe_layers = len(moe_layers)

    if num_moe_layers == 0:
        raise ValueError("No MoE layers found in frequencies.")

    # 确定专家数量（假设所有层的专家数一致）
    sample_layer = moe_layers[0]
    routed_data = frequencies[sample_layer]["routed"]
    num_routed_experts = len(routed_data)

    # 创建激活概率矩阵
    activation_matrix = np.zeros((num_moe_layers, num_routed_experts))

    # 填充矩阵
    for layer_idx, layer in enumerate(moe_layers):
        for expert_idx in range(num_routed_experts):
            activation_matrix[layer_idx, expert_idx] = frequencies[layer]["routed"][expert_idx]

    # 打印每层的激活总和（调试用）
    # print(f"Activation sum per layer: {activation_matrix.sum(axis=-1)}")

    # 设置绘图风格
    sns.set(style="whitegrid")

    # 创建热力图
    plt.figure(figsize=(base_width, base_height))
    ax = sns.heatmap(
        activation_matrix,
        cmap=cmap,
        annot=annot,
        fmt=".4f",
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5
    )
    ax.invert_yaxis()

    # 设置标题和轴标签
    plt.title(title, fontsize=16, pad=10)
    plt.xlabel("Expert Index", fontsize=12)
    plt.ylabel("Layer Index", fontsize=12)

    # 保存图形和激活矩阵
    if output_path is not None:
        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存热力图
        plt.savefig(output_path, format=output_format, dpi=dpi, bbox_inches=bbox_inches)

        # 保存激活矩阵为 JSON（与图片同名，后缀为 .json）
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump({
                "layers": moe_layers,
                "experts": list(range(num_routed_experts)),
                "activation_matrix": activation_matrix.tolist()
            }, f, indent=4)

    # 显示图形
    plt.show()

    return activation_matrix
    
    
def draw1(frequencies,save=False):
    # 假设 frequencies 是通过 model.get_all_expert_frequencies() 获取的
    

    # 确定 MoE 层数和专家数量
    num_moe_layers = len([layer_idx for layer_idx in frequencies.keys() if layer_idx != 0])  # 排除第 0 层
    num_routed_experts = 64  # DeepSeekV2-Lite 有 64 个路由专家

    # 创建激活概率矩阵
    activation_matrix = np.zeros((num_moe_layers, num_routed_experts))

    # 填充矩阵
    layer_counter = 0
    for layer_idx in sorted(frequencies.keys()):
        if layer_idx == 0:  # 跳过第 0 层（非 MoE 层）
            continue
        for expert_idx in range(num_routed_experts):
            activation_matrix[layer_counter, expert_idx] = frequencies[layer_idx][expert_idx]
        layer_counter += 1

    # 设置绘图风格
    sns.set(style="whitegrid")

    # 创建热力图
    plt.figure(figsize=(15, 10))  # 设置图形大小
    ax = sns.heatmap(
        activation_matrix,
        cmap="OrRd",  # 颜色映射：黄色到红色，值越大颜色越深
        annot=False,    # 不显示格子上的数值（格子太多会显得拥挤）
        fmt=".4f",      # 数值格式（如果 annot=True）
        # cbar_kws={'label': 'Percentage'},  # 颜色条标签
        # vmin=0,  
        # vmax=1,
        # annot=True,
        linewidths=.5
    )

    ax.invert_yaxis()

    # 设置标题和轴标签
    plt.title(f'Sentence Level', fontsize=16)
    plt.xlabel("Expert Index", fontsize=12)
    plt.ylabel("Layer Index", fontsize=12)
    if save:
        plt.savefig(f'/mnt/petrelfs/liuxinmin/moe/MoE-Infinity/moe_infinity/results/plot.svg', format="svg", dpi=300)
    # 显示图形
    plt.show()
    
# 定义一个递归函数，将 defaultdict 转换为普通字典
def convert_to_dict(nested_defaultdict):
    if isinstance(nested_defaultdict, defaultdict):
        # 递归处理每一层
        return {key: convert_to_dict(value) for key, value in nested_defaultdict.items()}
    else:
        # 如果不是 defaultdict，则直接返回值
        return nested_defaultdict



# 定义一个函数来执行相邻 text_idx 的与运算
def compute_adjacent_and(all_frequencies_dict):
    adjacent_and_results = {}

    # 获取所有 text_idx 并排序
    text_indices = sorted(all_frequencies_dict.keys())

    # 遍历相邻的 text_idx 对
    for i in range(len(text_indices) - 1):
        text_idx_1 = text_indices[i]
        text_idx_2 = text_indices[i + 1]

        # 初始化当前相邻对的结果
        adjacent_and_results[f"{text_idx_1}-{text_idx_2}"] = {}

        # 获取两个相邻 text_idx 的数据
        data_1 = all_frequencies_dict[text_idx_1]
        data_2 = all_frequencies_dict[text_idx_2]

        # 遍历每一层 layer_idx
        for layer_idx in set(data_1.keys()).union(data_2.keys()):
            experts_1 = data_1.get(layer_idx, {})
            experts_2 = data_2.get(layer_idx, {})

            # 对该层的所有 expert_idx 执行与运算
            adjacent_and_results[f"{text_idx_1}-{text_idx_2}"][layer_idx] = {
                expert_idx: experts_1.get(expert_idx, 0) & experts_2.get(expert_idx, 0)
                for expert_idx in set(experts_1.keys()).union(experts_2.keys())
            }

    return adjacent_and_results

def plot_layer_token_similarity_bar(
    activation_matrix,
    output_path=None,
    title="Average Cosine Similarity of Adjacent Requests per Layer",
    figsize=(10, 6),
    color='skyblue',
    edgecolor='black',
    output_format="svg",
    dpi=300,
    bbox_inches="tight"
):
    """
    绘制每个层中相邻 token 的平均专家激活相似度的柱状图，并保存 activation_matrix。

    参数：
        activation_matrix (np.ndarray): 形状为 (num_batch, num_layers, num_experts) 的激活矩阵
        output_path (str): 保存路径（不含后缀，例如 '/path/to/layer_token_similarity'）
        title (str): 图表标题
        figsize (tuple): 图表大小
        color (str): 柱子颜色
        edgecolor (str): 柱子边框颜色
        output_format (str): 图片格式（例如 'svg', 'png'）
        dpi (int): 保存图片的 DPI
        bbox_inches (str): 保存时的边界调整
    """
    # 检查输入维度
    if len(activation_matrix.shape) != 3:
        raise ValueError("activation_matrix must have shape (num_batch, num_layers, num_experts)")
    num_batch, num_layers, num_experts = activation_matrix.shape
    
    if num_batch < 2:
        raise ValueError("activation_matrix must have at least 2 tokens for adjacent comparison")

    # 计算每个层的相邻 token 平均余弦相似度
    vec_res = []
    for layer_idx in range(num_layers):
        # 获取当前层的激活数据
        layer_data = activation_matrix[:, layer_idx, :]  # [num_batch, num_experts]

        # 计算相邻 Request 的余弦相似度（向量化）
        tokens_prev = layer_data[:-1]  # [num_batch - 1, num_experts]
        tokens_next = layer_data[1:]   # [num_batch - 1, num_experts]
        dot_product = np.sum(tokens_prev * tokens_next, axis=1)
        norm_prev = np.linalg.norm(tokens_prev, axis=1)
        norm_next = np.linalg.norm(tokens_next, axis=1)
        similarities = dot_product / (norm_prev * norm_next + 1e-8)  # [num_batch - 1]

        # 计算平均相似度
        avg_similarity = np.mean(similarities)
        # print(f"Average Cosine Similarity for layer {layer_idx}: {avg_similarity}")
        vec_res.append(float(avg_similarity))

    # 绘制柱状图
    plt.figure(figsize=figsize)
    plt.bar(range(num_layers), vec_res, color=color, edgecolor=edgecolor)

    # 设置标题和标签
    plt.title(title, fontsize=14)
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Average Cosine Similarity", fontsize=12)

    # 设置 x 轴刻度
    plt.xticks(range(num_layers))
    plt.yticks(np.arange(0, 1.2, 0.2))

    # 添加网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 调整布局
    plt.tight_layout()

    if output_path is not None:
        # 保存图形和 activation_matrix
        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存柱状图
        plt.savefig(output_path.with_suffix(f".{output_format}"), format=output_format, dpi=dpi, bbox_inches=bbox_inches)

        # 保存 activation_matrix 和相似度为 JSON
        json_path = output_path.with_suffix(".json")
        with open(json_path, 'w') as f:
            json.dump({
                "batches": list(range(num_batch)),
                "layers": list(range(num_layers)),
                "experts": list(range(num_experts)),
                "activation_matrix": activation_matrix.tolist(),
                "average_cosine_similarities": vec_res
            }, f, indent=4)

    # 显示图形
    plt.show()