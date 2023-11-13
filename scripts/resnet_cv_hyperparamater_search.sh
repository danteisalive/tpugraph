#!/usr/bin/env bash

global_pooling_types=("mean" "max" "mean+max")
aggregation_types=("mean")
n_mp_layers=(2 3 4)
pre_hid_dims=(32 64)
gnn_hid_dims=(64 128)
gnn_out_dims=(64 128)

for nl in ${!n_mp_layers[@]}; do
    for at in ${!aggregation_types[@]}; do
        for gpt in ${!global_pooling_types[@]}; do
            for phd in ${!pre_hid_dims[@]}; do
                for ghd in ${!gnn_hid_dims[@]}; do    
                    for god in ${!gnn_out_dims[@]}; do

                        n_mp_layer=${n_mp_layers[$nl]}
                        aggregation_type=${aggregation_types[$at]}
                        global_pooling_type=${global_pooling_types[$gpt]}
                        pre_hid_dim=${pre_hid_dims[$phd]}
                        gnn_hid_dim=${gnn_hid_dims[$ghd]}
                        gnn_out_dim=${gnn_out_dims[$god]}
                        
                        echo "n_mp_layer $n_mp_layer, aggregation_type $aggregation_type, global_pooling_type $global_pooling_type, pre_hid_dim $pre_hid_dim, gnn_hid_dim $gnn_hid_dim, gnn_out_dim $gnn_out_dim," 

                        results_directory=$(pwd)/$(basename "$0")_results/

                        if [ ! -d "$results_directory" ]; then
                            mkdir -p "$results_directory"
                        fi

                        python ./../main_layout_resnet_cv.py \
                            --epochs 1 \
                            --num-configs 33 \
                            --batch-size 8 \
                            --num-splits 2 \
                            --device 'cuda' \
                            --num-gnn-layers $n_mp_layer \
                            --prenet-hidden-dim $pre_hid_dim \
                            --gnn-hidden-dim $gnn_hid_dim \
                            --gnn-out-dim $gnn_out_dim \
                            --aggr-type $aggregation_type \
                            --global-pooling-type $global_pooling_type \
                            --results-file-path ${results_directory}/resnet_cv.csv > \
                            ${results_directory}/resnet_cv_${n_mp_layer}_${aggregation_type}_${global_pooling_type}_${pre_hid_dim}_${gnn_hid_dim}_${gnn_out_dim}_$(date '+%Y-%m-%d_%H:%M:%S').log
                    done
                done
            done 
        done 
    done 
done 



