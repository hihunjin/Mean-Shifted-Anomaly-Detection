# cifar10
# waterbirds
# color_mnist
export DS="
celeba
"
# two_class_color_mnist

export backbones="
clip
18
"

# False
export is_angulars="
True
"
for ds in $DS
do
    for backbone in $backbones
    do
        if [ $ds = "two_class_color_mnist" ]
        then
            export target_indices="
            0
            1
            "
        elif [ $ds = "color_mnist" ]
        then
            export target_indices="
            0
            1
            "
        elif [ $ds = "waterbirds" ]
        then
            export target_indices="
            0
            1
            "
        elif  [ $ds = "celeba" ]
        then
            export target_indices="
            9
            15
            39
            "
        fi
        for target_index in $target_indices
        do
            for is_angular in $is_angulars
            do
                echo ds=$ds backbone=$backbone target_index=$target_index is_angular=$is_angular
                echo "sudo -H -E PYTHONPATH=$PYTHONPATH:$pwd CUDA_VISIBLE_DEVICES=1 python main.py --dataset=$ds --backbone $backbone --target_index $target_index --angular $is_angular"
                # sudo -H -E PYTHONPATH=$PYTHONPATH:$pwd \
                # CUDA_VISIBLE_DEVICES=1 \
                # python main.py \
                # --dataset=$ds --backbone $backbone \
                # --target_index $target_index --angular $is_angular
            done
        done
    done
done