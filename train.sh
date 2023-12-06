#!/bin/bash 

#echo "Allocating GPU"
#source ~/gpu_alloc.sh

script_dir=$(pwd)
echo $script_dir

#exit 0

show_help()
{
    # display help
    echo "Usage: $0 -m [model name]"
    echo "options:"
    echo "  -m [model name]    Specify model name to execute. valid model names [vanilla, lora, amp, grad_acc, ckpt, test]"
    echo "  -h                 Display this help message"
    #exit 1
    #return 1
}

model_option=""

# check if no arguments are provided
if [ "$#" -eq 0 ]; then
    echo "Error: No arguments provided. Use -h for help."
    #exit 1
    #return 1
fi

#parse command line arguments
while getopts "m:h" opt; do
    case $opt in
        m)
            model_option="$OPTARG"
            ;;
        h)
            show_help
            ;;
        \?)
            echo "Invalid option: -$OPTARG"
            show_help
            ;;
        :)
            echo "Option -$OPTARG requires an argument."
            show_help
            ;;
    esac
done

# check if model name is provided
if [ -z "$model_option" ]; then
    echo "Error: Model name is required. Use -h for help."
    #exit 1
    #return 1
fi

model_name="$model_option"

# execute script based on model name
case $model_name in
    "vanilla")
        cd $script_dir/llama
        ln -sf model_vanilla.py model.py
        sed -i '94s/.*/        model_args.n_layers = 8/g' generation.py
        cd $script_dir
        python $script_dir/train.py
        ;;
    "lora")
        cd $script_dir/llama
        ln -sf model_lora.py model.py
        sed -i '94s/.*/        model_args.n_layers = 32/g' generation.py
        cd $script_dir
        python $script_dir/lora_train.py
        ;;
    "amp")
        cd $script_dir/llama
        ln -sf model_lora.py model.py
        sed -i '94s/.*/        model_args.n_layers = 24/g' generation.py
        cd $script_dir
        python $script_dir/amp_lora_train.py
        ;;
    "grad_acc")
        cd $script_dir/llama
        ln -sf model_lora.py model.py
        sed -i '94s/.*/        model_args.n_layers = 24/g' generation.py
        cd $script_dir
        python $script_dir/grad_acc_train.py
        ;;
    "ckpt")
        cd $script_dir/llama
        ln -sf model_lora_ckpt.py model.py
        sed -i '94s/.*/        #model_args.n_layers = 32/g' generation.py
        cd $script_dir
        python $script_dir/grad_acc_train.py
        ;;
    *)
        echo "Error: Unknown model name. Use -h for help."
        #exit 1
        #return 1
        ;;
esac

#exit 0
