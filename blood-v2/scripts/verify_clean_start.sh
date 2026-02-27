#!/bin/bash
# 验证是否真的从头开始训练

echo "检查训练目录..."
if [ -d "train_dir/blood_v2_warmup" ]; then
    echo "❌ train_dir/blood_v2_warmup 存在！"
    echo "   最后修改时间:"
    ls -lh train_dir/blood_v2_warmup | head -5
    exit 1
fi

echo "检查checkpoint目录..."
if [ -d "checkpoints/blood_v2_warmup" ]; then
    echo "❌ checkpoints/blood_v2_warmup 存在！"
    ls -lh checkpoints/blood_v2_warmup | head -5
    exit 1
fi

if [ -d "checkpoints/league" ]; then
    count=$(find checkpoints/league -type f | wc -l)
    if [ $count -gt 0 ]; then
        echo "❌ checkpoints/league 包含 $count 个文件！"
        ls -lh checkpoints/league | head -5
        exit 1
    fi
fi

echo "✅ 所有目录已清空，可以开始全新训练"
echo ""
echo "建议的清理命令:"
echo "  rm -rf train_dir/blood_v2_*"
echo "  rm -rf checkpoints/blood_v2_*"
echo "  rm -rf checkpoints/league/*"