# # 全 token 平均（默认）
# python moti_adjacent_frame_hidden_similarity.py -l 0

# # 仅 CLS
# python moti_adjacent_frame_hidden_similarity.py -l 0 --only_cls

# # 全 token 平均（默认）
# python moti_adjacent_frame_hidden_similarity.py -l 5

# # 仅 CLS
# python moti_adjacent_frame_hidden_similarity.py -l 5 --only_cls

# # 全 token 平均（默认）
# python moti_adjacent_frame_hidden_similarity.py -l 11

# # 仅 CLS
# python moti_adjacent_frame_hidden_similarity.py -l 11 --only_cls

# # 全 token 平均（默认）
# python moti_adjacent_frame_hidden_similarity.py -l 0 --frame_interval 1

# # 仅 CLS
# python moti_adjacent_frame_hidden_similarity.py -l 0 --frame_interval 1 --only_cls

# # 全 token 平均（默认）
# python moti_adjacent_frame_hidden_similarity.py -l 5 --frame_interval 1

# # 仅 CLS
# python moti_adjacent_frame_hidden_similarity.py -l 5 --frame_interval 1 --only_cls

# # 全 token 平均（默认）
# python moti_adjacent_frame_hidden_similarity.py -l 11 --frame_interval 1

# # 仅 CLS
# python moti_adjacent_frame_hidden_similarity.py -l 11 --frame_interval 1 --only_cls



# # 只画 layer 0
# python motivation_results/adjacent_frame/show_figure.py -l 0 --x_label second

# # 只画 frame_interval=1 且 cls
# python motivation_results/adjacent_frame/show_figure.py --frame_interval 1 --x_label second

python motivation_results/adjacent_frame/show_figure.py -l 0 --x_label second --only_cls

# # 横轴为秒
# python motivation_results/adjacent_frame/show_figure.py -l 6 --x_label second

# # 组合使用
# python motivation_results/adjacent_frame/show_figure.py -l 11 --frame_interval 10 --only_cls --x_label second -o my_plot.png