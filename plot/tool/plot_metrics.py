import json
import matplotlib.pyplot as plt
import os

def load_metrics(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', {})
    
    gsm8k_match = results.get('gsm8k', {}).get('exact_match,strict-match', 0)
    humaneval_pass1 = results.get('humaneval', {}).get('pass@1,create_test', 0)
    
    return gsm8k_match, humaneval_pass1

def plot_metrics(files):
    labels = ["GSM8K (exact_match)", "HumanEval (pass@1)"]
    
    file_labels = []
    gsm8k_scores = []
    humaneval_scores = []
    
    for idx, f in enumerate(files):
        filename = os.path.basename(f)
        try:
            m1, m2 = load_metrics(f)
            if '_sp' in filename:
                sp_val = filename.split('_sp')[1].split('_')[0]
                label = f"Sparsity {float(sp_val)*100:.0f}%"
            elif 'none' in filename:
                label = "No Sparsity"
            else:
                label = f"Model {idx+1}"
            file_labels.append(label)
            gsm8k_scores.append(m1)
            humaneval_scores.append(m2)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    x = [0, 1]
    num_files = len(file_labels)
    total_width = 0.8
    width = total_width / max(1, num_files)

    # A color-blind friendly and publication-ready palette (Seaborn deep inspired)
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD']

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Optional: add a light horizontal grid for better readability
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.3)
    ax.set_axisbelow(True) # put grid behind bars
    
    for i in range(num_files):
        offset = (i - num_files / 2.0 + 0.5) * width
        c = colors[i % len(colors)]
        rects = ax.bar([pos + offset for pos in x], [gsm8k_scores[i], humaneval_scores[i]], 
                       width, label=file_labels[i], color=c, edgecolor='white', linewidth=1.2)
        ax.bar_label(rects, fmt='%.3f', padding=3, fontsize=10)

    ax.set_ylabel('Scores', fontsize=12, fontweight='bold')
    ax.set_title('Evaluation Metrics across Sparsity Levels', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    
    # Place the legend inside the plot area with a bounding box
    ax.legend(fontsize='10', 
              loc='best', frameon=True, edgecolor='black', fancybox=True)

    fig.tight_layout()
    output_dir = '/home/pairshoe/lxm_flash/On-Device-MoE/plot/result'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'metrics_plot.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    # The user provided the same file name twice, but you can change these to point to the real files
    dir_path = "/home/pairshoe/lxm_flash/On-Device-MoE/Sparsity_eval/result/deepseek_v2_lite_chat/"
    files = [
        dir_path + "DeepSeek_V2_Lite_Chat_gsm8k_humaneval_none_5shot.json",
        dir_path + "DeepSeek_V2_Lite_Chat_gsm8k_humaneval_hybrid_sp0.1_ep1.0_5shot.json",
        dir_path + "DeepSeek_V2_Lite_Chat_gsm8k_humaneval_hybrid_sp0.2_ep1.0_5shot.json",
        dir_path + "DeepSeek_V2_Lite_Chat_gsm8k_humaneval_hybrid_sp0.3_ep1.0_5shot.json",
        dir_path + "DeepSeek_V2_Lite_Chat_gsm8k_humaneval_hybrid_sp0.4_ep1.0_5shot.json",
        dir_path + "DeepSeek_V2_Lite_Chat_gsm8k_humaneval_hybrid_sp0.5_ep1.0_5shot.json"
    ]
    plot_metrics(files)
