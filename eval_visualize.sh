result_file="benchmark_results/useful/benchmark_Video-MME_medium_20260309_180348.json"
# 指定结果文件和输出路径
python eval_videomme_and_visualize.py --results_file $result_file \
    --video_duration_type medium \
    --return_categories_accuracy \
    --return_sub_categories_accuracy \
    --return_task_types_accuracy \
    --output motivation_results/videomme_medium_bar.png

result_file="benchmark_results/useful/benchmark_Video-MME_short_20260309_170326.json"
# 指定结果文件和输出路径
python eval_videomme_and_visualize.py --results_file $result_file \
    --video_duration_type short \
    --return_categories_accuracy \
    --return_sub_categories_accuracy \
    --return_task_types_accuracy \
    --output motivation_results/videomme_short_bar.png
