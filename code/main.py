from method import *
from result import *


# ----------------参数设置区----------------
data_path = 'D:/桌面/F/2024.05.10-超模体韧性/data/'
result_path = 'D:/桌面/F/2024.05.10-超模体韧性/result_fig/'
save_path = 'D:/桌面/F/2024.05.10-超模体韧性/result_fig/'


# ----------------结果计算----------------
# 下载、清洗并校验报告中的 59 个 A 级现实世界超图；已完成数据会自动跳过
# process_all_grade_a_real_hypergraphs(data_path)

# Random hypergraph：para 为 edge_factor=M/N，每组生成20个具有默认混合超边大小分布的网络
# artifical_hypergraph_gen(data_path, 'r', num = 20, para = 0.5) # 生成稀疏随机超图，超边数约为节点数的0.5倍
# artifical_hypergraph_gen(data_path, 'r', num = 20, para = 1.0) # 生成基准随机超图，超边数约等于节点数
# artifical_hypergraph_gen(data_path, 'r', num = 20, para = 2.0) # 生成中等密度随机超图，超边数约为节点数的2倍
# artifical_hypergraph_gen(data_path, 'r', num = 20, para = 4.0) # 生成较密随机超图，超边数约为节点数的4倍

# Small-world hypergraph：para 为重连概率p，默认degree=4，每组生成20个网络
# artifical_hypergraph_gen(data_path, 'sw', num = 20, para = 0.01) # 生成极低重连概率的小世界超图
# artifical_hypergraph_gen(data_path, 'sw', num = 20, para = 0.05) # 生成低重连概率的小世界超图
# artifical_hypergraph_gen(data_path, 'sw', num = 20, para = 0.1) # 生成典型中低重连概率的小世界超图
# artifical_hypergraph_gen(data_path, 'sw', num = 20, para = 0.5) # 生成高重连概率的小世界超图

# Preferential-attachment hypergraph：para 为节点到达概率p，每组生成20个网络
# artifical_hypergraph_gen(data_path, 'ba', num = 20, para = 0.1) # 生成低节点到达率、以新增超边事件为主的无标度超图
# artifical_hypergraph_gen(data_path, 'ba', num = 20, para = 0.2) # 生成低节点到达率、以新增超边事件为主的无标度超图
# artifical_hypergraph_gen(data_path, 'ba', num = 20, para = 0.4) # 生成节点到达与超边到达相对均衡的无标度超图
# artifical_hypergraph_gen(data_path, 'ba', num = 20, para = 0.8) # 生成高节点到达率的无标度超图

# Hyper-motif 分析参数：r=2 精确统计，r=3 使用分层 wedge 重要性抽样
r_values = (2, 3)
maximum_cardinality = 6
random_seed = 2026

# # 1. 生成保持节点度与逐边 cardinality 的 100 个 H_m(d) null models
# hypergraph_null_model_cal(data_path, result_path, null_num = 100, seed = random_seed, chain_num = 4)

# # 2. 统计原始网络 hyper-motifs；r=3 每个网络抽样 200000 次
# hypergraph_hypermotifs_sta_cal(data_path, result_path, r_values = r_values, t = maximum_cardinality, r3_sample_num = 200000, seed = random_seed)

# # 3. 统计 100 个 null models 的 hyper-motifs；每个 null 的 r=3 抽样 50000 次
# hypergraph_null_model_hypermotifs_sta_cal(data_path, result_path, r_values = r_values, t = maximum_cardinality, r3_sample_num = 50000, seed = random_seed)

# # 4. 分别在每个 r 内计算 C_i
# hypergraph_ci_cal(data_path, result_path, r_values = r_values, t = maximum_cardinality, seed = random_seed)

# 5. 人工网络计算对应生成模型理论；现实网络按自身 N、M 与 cardinality inventory
#    分别计算 random、small-world 和 BA 三套 connected、unlabeled 理论概率
# hypergraph_ci_theory_cal(data_path, result_path, r_values = r_values, t = maximum_cardinality, theory_sample_num = 200000, seed = random_seed)

# 6. 使用 100 个 null models 和 epsilon=4 计算 Delta_i
# hypergraph_deltai_cal(data_path, result_path, r_values = r_values, t = maximum_cardinality, epsilon = 4.0, seed = random_seed)

# 7. 验证实际 C_i 与理论分布的符合程度
# hypergraph_ci_theory_validate(data_path, result_path, r_values = r_values, t = maximum_cardinality, bootstrap_num = 1000, seed = random_seed)

# 8. 计算人工与现实网络分布的显著性；没有现实数据时记录为 insufficient
# hypergraph_ar_significance_cal(data_path, result_path, r_values = r_values, t = maximum_cardinality)

# 9. 给定N，M和Delta_i情况下的超图生成模型
# for i, n in enumerate(np.random.randint(low = 50, high = 1000, size = 50)):
#     print((i + 1, 50, n))
#     hypergraph_generation_model_N_M_delta(save_path, mode = 'random', N = n)

# 10. 读取现实网络的已有Delta_i，筛选N<max_node_lim并重构，结果保存到10号文件夹
# hypergraph_generation_model_N_M_delta_from_real_data(data_path, result_path, save_path, max_node_lim = 1000)

# 11-13. 现实网络重构前后分布比较，读取10号结果并分别保存到11、12、13号文件夹
# real_generation_degree_comparasion(data_path, save_path)
# real_generation_cardinality_comparasion(data_path, save_path)
# real_generation_ci_comparasion(data_path, save_path)

# 15. SI：整理9/10号全部重构结果的r=2/3输入与重构Delta，不重新生成网络
# hypergraph_generation_deltai_comparasion(save_path)

# 18. 从9/10号原始结果计算重构精度，输出Excel到18.reconstruction_sta_cal。
# 不重新生成网络；完整保留canonical motif编号，默认active阈值为0.1。
if __name__ == '__main__':
    reconstruction_accuracy = hypergraph_reconstruction_sta_cal(data_path, save_path)

# ----------------图片绘制----------------
# fig_artificial_ci_distribution(result_path, save_path, r_values = r_values, axes_scale = [0, 0, 0.3, 0.12])
# fig_artificial_deltai_distribution(result_path, save_path, r_values = r_values, axes_scale = [0, 0, 0.3, 0.12])
# fig_real_ci_distribution(result_path, save_path, r_values = r_values, axes_scale = [0, 0, 0.3, 0.12])
# fig_real_deltai_distribution(result_path, save_path, r_values = r_values, axes_scale = [0, 0, 0.3, 0.12])

# fig_ci_real_theory_probability(result_path, save_path, r_values = r_values)

# fig_artificial_ci_deltai_joint(result_path, save_path, r_values = r_values, axes_scale = [0, 0, 0.4, 0.15])
# fig_real_ci_deltai_joint(result_path, save_path, r_values = r_values, axes_scale = [0, 0, 0.4, 0.15])
# fig_ci_significance(result_path, save_path, r_values = r_values)
# fig_deltai_significance(result_path, save_path, r_values = r_values)

# fig_real_net_ci_deltai_cluster_heatmap(result_path, save_path, r_values = r_values)

# 第三部分：panel b/c 共用同一个10号重构结果；修改此文件名即可切换网络。
net_name = '1.26_54_real_3_6.pkl'
# fig_hypergraph_reconstruction_structure_comparasion(data_path, save_path, net_name, axes_scale = [0, 0, 0.4, 0.4])
# fig_hypergraph_reconstruction_deltai_comparasion(save_path, net_name, axes_scale = [0, 0, 0.4, 0.1],before_color = '#777777', after_color = '#D64B4B')
# fig_artificial_deltai_scatter(save_path, axes_scale = [0, 0, 0.3, 0.3], r_values = r_values,marker_size = 6, alpha = 0.1)
# fig_real_deltai_scatter(save_path, axes_scale = [0, 0, 0.3, 0.3], r_values = r_values,marker_size = 6, alpha = 0.1)
# fig_error_show(save_path, axes_scale = [0, 0, 0.2, 0.3])

# SI：16号现实网络分布图；17号全部网络Delta双面板图
# fig_real_generation_degree_comparasion(save_path, axes_scale = [0, 0, 0.6, 0.6])
# fig_real_generation_cardinality_comparasion(save_path, axes_scale = [0, 0, 0.6, 0.6])
# fig_real_generation_ci_comparasion(save_path, axes_scale = [0, 0, 0.6, 0.6])
# fig_real_generation_distribution_comparasion(save_path, axes_scale = [0, 0, 0.6, 0.6])
# fig_hypergraph_generation_deltai_comparasion(save_path, axes_scale = [0, 0, 0.6, 0.6])
