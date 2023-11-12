#!/usr/bin/env bash

global_pooling_types=("mean" "max" "mean-max")
aggregation_types=("sum" "mean" "max")
n_layers=(2 3 4)


for nl in ${!n_layers[@]}; do
    for at in ${!aggregation_types[@]}; do
        for gpt in ${!global_pooling_types[@]}; do
            
            n_layer=${n_layers[$nl]}
            aggregation_type=${aggregation_types[$at]}
            global_pooling_type=${global_pooling_types[$gpt]}
            
            echo "n_layers $n_layer, aggregation_type $aggregation_type global_pooling_type $global_pooling_type" 

            results_directory=$(pwd)/$(basename "$0")_results/

            if [ ! -d "$results_directory" ]; then
                mkdir -p "$results_directory"
            fi

            ./../main_layout_resnet_cv.py \
                --epochs 50 \
                --num-confgis 33 \
                --batch-size 8 \ 
                --num-splits 5 \
                --num-layers $n_layer \
                --aggr-type $aggregation_type \
                --global-pooling-type $global_pooling_type \
                --results-file-path ${results_directory}/resnet_cv_$(date '+%Y-%m-%d_%H:%M:%S').csv

        done 
    done 
done 







for i in ${!ranges[@]}; do
    range=${ranges[$i]}
    max_num_batched_tokens=${max_batch_total_tokens_vals[$i]}
    echo "range $range, max_num_batched_tokens $max_num_batched_tokens"

    start_model_server $max_num_batched_tokens

    pushd ..

        results_directory=$(pwd)/$(basename "$0")_results/

        if [ ! -d "$results_directory" ]; then
            mkdir -p "$results_directory"

        fi
        ./benchmark_throughput.py \
            --port $PORT \
            --backend vLLM  \
            --random_prompt_lens_mean 512 \
            --random_prompt_lens_range 0 \
            --random_prompt_count 30 \
            --gen_random_prompts \
            --fixed_max_tokens $range \
            --results_filename ${results_directory}/vllm_test_${range}_$(date '+%Y-%m-%d_%H:%M:%S').log
    popd
    kill_model_server

done
