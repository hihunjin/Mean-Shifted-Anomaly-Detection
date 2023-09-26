# cifar10
# two_class_color_mnist
# waterbirds
export DS="
celeba
"
export backbones="
clip
18
"

export is_angulars="
False
True
"
for ds in $DS
do
    for backbone in $backbones
    do
        # if [ $ds = "two_class_color_mnist" ]
        # then
        #     export target_indices="
        #     0
        #     1
        #     "
        # elif [ $ds = "waterbirds" ]
        # then
        #     export target_indices="
        #     0
        #     1
        #     "
            # 22
        if  [ $ds = "celeba" ]
        then
            export target_indices="
            15
            39
            "
        fi
        for target_index in $target_indices
        do
            for is_angular in $is_angulars
            do
                echo ds=$ds backbone=$backbone target_index=$target_index is_angular=$is_angular
                PYTHONPATH=$PYTHONPATH:$pwd python main.py --dataset=$ds --backbone backbone --target_index $target_index \
                --angular=$is_angular
            done
        done
    done
done