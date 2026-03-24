result_file_short="benchmark_results/venus/benchmark_Video-MME_short_20260309_170326.json"
python eval_videomme.py --results_file $result_file_short\
    --video_duration_type short \
    --return_categories_accuracy \
    --return_sub_categories_accuracy \
    --return_task_types_accuracy

# result_file_medium="/home/ubuntu/liangh/Project-Video/HelloWorld/benchmark_results/useful/benchmark_Video-MME_medium_20260309_180348.json"
# python eval_videomme.py --results_file $result_file_medium\
#     --video_duration_type medium \
#     --return_categories_accuracy \
#     --return_sub_categories_accuracy \
#     --return_task_types_accuracy
echo "hahahahahhahahahahhahahaha"
result_file_short="benchmark_results/symphony/benchmark_Video-MME_short_20260319_153714.json"
python eval_videomme.py --results_file $result_file_short\
    --video_duration_type short \
    --return_categories_accuracy \
    --return_sub_categories_accuracy \
    --return_task_types_accuracy