export PYTHONPATH=/home/yueyulin/github/CosyVoice:/home/yueyulin/github/CosyVoice/third_party/Matcha-TTS/:/home/yueyulin/github/RWKVTTS

# 设置默认参数
LANGUAGE="zh"
OUTPUT_DIR="/home/yueyulin/data/speech_corpus"
COSY_MODEL_DIR="/home/yueyulin/models/CosyVoice2-0.5B/"
PROMPTS_DIR="extract_data/prompts/zh"
DEVICE="cuda:0"
PARQUET_FILES=()
JSONL_FILES=()
FILE_TYPE="" # 用于标记文件类型
is_cross_lingual=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --language)
      LANGUAGE="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --cosy_model_dir)
      COSY_MODEL_DIR="$2"
      shift 2
      ;;
    --prompts_dir)
      PROMPTS_DIR="$2"
      shift 2
      ;;
    --parquet_files)
      # 接收多个parquet文件路径
      shift
      while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
        PARQUET_FILES+=("$1")
        shift
      done
      FILE_TYPE="parquet"
      ;;
    --jsonl_files)
      # 接收多个jsonl文件路径
      shift
      while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
        JSONL_FILES+=("$1")
        shift
      done
      FILE_TYPE="jsonl"
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --cross_lingual)
      is_cross_lingual="--is_cross_lingual"
      shift
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

# 检查是否提供了文件
if [ "$FILE_TYPE" == "parquet" ]; then
  if [ ${#PARQUET_FILES[@]} -eq 0 ]; then
    echo "错误: 未指定parquet文件，请使用 --parquet_files 参数"
    exit 1
  fi
  FILES=("${PARQUET_FILES[@]}")
  FILE_ARG="--parquet_files"
  echo "将处理 ${#FILES[@]} 个parquet文件"
elif [ "$FILE_TYPE" == "jsonl" ]; then
  if [ ${#JSONL_FILES[@]} -eq 0 ]; then
    echo "错误: 未指定jsonl文件，请使用 --jsonl_files 参数"
    exit 1
  fi
  FILES=("${JSONL_FILES[@]}")
  FILE_ARG="--jsonl_files"
  echo "将处理 ${#FILES[@]} 个jsonl文件"
else
  echo "错误: 请使用 --parquet_files 或 --jsonl_files 参数指定输入文件"
  exit 1
fi

echo "运行参数:"
echo "语言: $LANGUAGE"
echo "输出目录: $OUTPUT_DIR"
echo "模型目录: $COSY_MODEL_DIR"
echo "提示词目录: $PROMPTS_DIR"
echo "设备: $DEVICE"
echo "文件类型: $FILE_TYPE"

# 确保输出目录存在
mkdir -p $OUTPUT_DIR

# 启动处理进程，每个文件一个进程
for ((i=0; i<${#FILES[@]}; i++)); do
  FILE="${FILES[$i]}"
  FILENAME=$(basename "$FILE")
  
  echo "处理文件 $FILENAME 使用 $DEVICE"
  
  # 在后台启动进程
  nohup python data/utils/utilitie.py \
    --task generate_speech_tokens \
    --language $LANGUAGE \
    $is_cross_lingual \
    $FILE_ARG "$FILE" \
    --output_dir $OUTPUT_DIR \
    --cosy_model_dir $COSY_MODEL_DIR \
    --prompts_dir $PROMPTS_DIR \
    --device "$DEVICE" > "$OUTPUT_DIR/log_${FILENAME%.*}.log" 2>&1 &
  
  # 记录进程ID
  PID=$!
  echo "启动进程 PID: $PID 处理文件: $FILENAME 使用 $DEVICE"
  
  # 等待一点时间确保进程启动
  sleep 5
done

echo "所有处理进程已启动，日志文件保存在 $OUTPUT_DIR 目录"
echo "使用 'ps aux | grep utilitie.py' 命令查看运行状态"
echo "使用 'nvidia-smi' 命令监控GPU使用情况"

# 等待所有后台进程完成
wait
echo "所有处理已完成"