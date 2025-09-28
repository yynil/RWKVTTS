#! /bin/bash

# 默认值
xy_tokenizer_config_path=""
xy_tokenizer_ckpt_path=""
num_proc=4
num_output_files=4
from_index=0
to_index=""

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -i, --input_dir DIR      输入目录 (必需)"
    echo "  -o, --output_dir DIR     输出目录 (必需)"
    echo "  -c, --config_path PATH   XY_Tokenizer配置文件路径 (必需)"
    echo "  -k, --ckpt_path PATH     XY_Tokenizer检查点文件路径 (必需)"
    echo "  -p, --num_proc NUM       进程数 (默认: $num_proc)"
    echo "  -n, --num_output_files NUM 生成的shell文件数量 (默认: $num_output_files)"
    echo "  -f, --from_index NUM     起始索引 (默认: $from_index)"
    echo "  -t, --to_index NUM       结束索引 (默认: 处理所有文件)"
    echo "  -h, --help               显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -i /path/to/input -o /path/to/output -c /path/to/config.yaml -k /path/to/checkpoint.ckpt"
    echo "  $0 --input_dir /path/to/input --output_dir /path/to/output --config_path /path/to/config.yaml --ckpt_path /path/to/checkpoint.ckpt --num_proc 8 --num_output_files 4"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input_dir)
            input_dir="$2"
            shift 2
            ;;
        -o|--output_dir)
            output_dir="$2"
            shift 2
            ;;
        -c|--config_path)
            xy_tokenizer_config_path="$2"
            shift 2
            ;;
        -k|--ckpt_path)
            xy_tokenizer_ckpt_path="$2"
            shift 2
            ;;
        -p|--num_proc)
            num_proc="$2"
            shift 2
            ;;
        -n|--num_output_files)
            num_output_files="$2"
            shift 2
            ;;
        -f|--from_index)
            from_index="$2"
            shift 2
            ;;
        -t|--to_index)
            to_index="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$input_dir" ] || [ -z "$output_dir" ] || [ -z "$xy_tokenizer_config_path" ] || [ -z "$xy_tokenizer_ckpt_path" ]; then
    echo "错误: 必须提供 input_dir、output_dir、config_path 和 ckpt_path"
    show_help
    exit 1
fi

# 检查目录和文件是否存在
if [ ! -d "$input_dir" ]; then
    echo "错误: 输入目录 '$input_dir' 不存在"
    exit 1
fi

if [ ! -f "$xy_tokenizer_config_path" ]; then
    echo "错误: XY_Tokenizer配置文件 '$xy_tokenizer_config_path' 不存在"
    exit 1
fi

if [ ! -f "$xy_tokenizer_ckpt_path" ]; then
    echo "错误: XY_Tokenizer检查点文件 '$xy_tokenizer_ckpt_path' 不存在"
    exit 1
fi

echo "=== 参数设置 ==="
echo "输入目录: $input_dir"
echo "输出目录: $output_dir"
echo "XY_Tokenizer配置文件: $xy_tokenizer_config_path"
echo "XY_Tokenizer检查点文件: $xy_tokenizer_ckpt_path"
echo "进程数: $num_proc"
echo "输出文件数: $num_output_files"
echo "起始索引: $from_index"
if [ -n "$to_index" ]; then
    echo "结束索引: $to_index"
else
    echo "结束索引: 处理所有文件"
fi
echo ""

echo "=== 1. 列出 input_dir 下的所有子目录 ==="
input_subdirs=()
if [ -z "$(ls -A "$input_dir")" ]; then
    echo "input_dir 是空目录"
else
    for item in "$input_dir"/*; do
        if [ -d "$item" ]; then
            dirname=$(basename "$item")
            input_subdirs+=("$dirname")
            echo "  - $dirname"
        fi
    done
fi

echo ""
echo "=== 2. 列出 output_dir 下的所有子目录 ==="
output_subdirs=()
if [ ! -d "$output_dir" ]; then
    echo "输出目录不存在，将自动创建"
else
    if [ -z "$(ls -A "$output_dir")" ]; then
        echo "output_dir 是空目录"
    else
        for item in "$output_dir"/*; do
            if [ -d "$item" ]; then
                dirname=$(basename "$item")
                output_subdirs+=("$dirname")
                echo "  - $dirname"
            fi
        done
    fi
fi

echo ""
echo "=== 3. 找出需要完成的目录（在input_dir中存在但在output_dir中不存在）==="
pending_dirs=()

for input_subdir in "${input_subdirs[@]}"; do
    found=false
    for output_subdir in "${output_subdirs[@]}"; do
        if [ "$input_subdir" = "$output_subdir" ]; then
            found=true
            break
        fi
    done
    
    if [ "$found" = false ]; then
        pending_dirs+=("$input_subdir")
    fi
done

if [ ${#pending_dirs[@]} -eq 0 ]; then
    echo "所有目录都已完成！"
    exit 0
else
    echo "需要完成的目录："
    for dir in "${pending_dirs[@]}"; do
        echo "  - $dir"
    done
    echo ""
    echo "总共需要完成 ${#pending_dirs[@]} 个目录"
    
    echo ""
    echo "=== 4. 生成shell文件 ==="
    
    # 计算每个文件包含的命令数量
    total_dirs=${#pending_dirs[@]}
    commands_per_file=$(( (total_dirs + num_output_files - 1) / num_output_files ))
    
    echo "每个shell文件将包含约 $commands_per_file 个命令"
    echo ""
    
    # 生成shell文件
    for ((i=0; i<num_output_files; i++)); do
        filename="extract_xy_tokens_shell_$((i+1)).sh"
        
        # 创建shell文件头部
        cat > "$filename" << EOF
#! /bin/bash

# 自动生成的脚本 - 文件 $((i+1))/$num_output_files
# 生成时间: $(date)
# 输入目录: $input_dir
# 输出目录: $output_dir
# XY_Tokenizer配置文件: $xy_tokenizer_config_path
# XY_Tokenizer检查点文件: $xy_tokenizer_ckpt_path
# 进程数: $num_proc
# 起始索引: $from_index
# 结束索引: $to_index

# 设置PYTHONPATH
export PYTHONPATH=\$(pwd):\$(pwd)/third_party:\$PYTHONPATH
echo "设置PYTHONPATH: \$PYTHONPATH"

echo "开始执行文件 $filename"
echo "时间: \$(date)"
echo ""

EOF
        
        # 计算当前文件应该包含的目录范围
        start_idx=$((i * commands_per_file))
        end_idx=$((start_idx + commands_per_file - 1))
        
        # 确保不超出数组范围
        if [ $end_idx -ge $total_dirs ]; then
            end_idx=$((total_dirs - 1))
        fi
        
        # 如果start_idx超出范围，说明没有更多命令了
        if [ $start_idx -ge $total_dirs ]; then
            echo "# 没有更多命令需要执行" >> "$filename"
        else
            # 添加命令到文件
            for ((j=start_idx; j<=end_idx; j++)); do
                dir="${pending_dirs[j]}"
                # 处理路径，避免双斜杠
                input_path=$(echo "$input_dir/$dir" | sed 's|//*|/|g')
                output_path=$(echo "$output_dir/$dir" | sed 's|//*|/|g')
                
                # 构建命令
                cmd="python utils/extract_xy_tokens.py --input_dir $input_path --output_dir $output_path --xy_tokenizer_config_path $xy_tokenizer_config_path --xy_tokenizer_ckpt_path $xy_tokenizer_ckpt_path --num_proc $num_proc --from_index $from_index"
                
                # 如果指定了结束索引，添加到命令中
                if [ -n "$to_index" ]; then
                    cmd="$cmd --to_index $to_index"
                fi
                
                echo "echo \"处理目录: $dir\"" >> "$filename"
                echo "$cmd" >> "$filename"
                echo "echo \"完成目录: $dir\"" >> "$filename"
                echo "" >> "$filename"
            done
        fi
        
        # 添加文件尾部
        cat >> "$filename" << EOF
echo ""
echo "文件 $filename 执行完成"
echo "时间: \$(date)"
EOF
        
        # 设置执行权限
        chmod +x "$filename"
        
        echo "生成文件: $filename"
        if [ $start_idx -le $end_idx ] && [ $start_idx -lt $total_dirs ]; then
            echo "  包含目录: ${pending_dirs[start_idx]} 到 ${pending_dirs[end_idx]}"
        else
            echo "  (空文件)"
        fi
    done
    
    echo ""
    echo "=== 5. 运行说明 ==="
    echo "已生成 $num_output_files 个shell文件，你可以在不同的终端中并行运行："
    echo ""
    for ((i=1; i<=num_output_files; i++)); do
        echo "终端 $i: ./extract_xy_tokens_shell_$i.sh"
    done
    echo ""
    echo "或者使用后台运行："
    for ((i=1; i<=num_output_files; i++)); do
        echo "nohup ./extract_xy_tokens_shell_$i.sh > log_xy_$i.out 2>&1 &"
    done
fi 