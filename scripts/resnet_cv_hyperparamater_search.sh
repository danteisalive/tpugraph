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
                --epochs 1 \
                --num-confgis 33 \
                --batch-size 8 \ 
                --num-splits 5 \
                --num-layers $n_layer \
                --aggr-type $aggregation_type \
                --global-pooling-type $global_pooling_type \
                --results-file-path ${results_directory}/resnet_cv.csv > ${results_directory}/resnet_cv_${n_layer}_${aggregation_type}_${global_pooling_type}_$(date '+%Y-%m-%d_%H:%M:%S').log

        done 
    done 
done 



