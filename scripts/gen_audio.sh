#!/bin/bash
set -e

# Voice to use (zh_CN)
VOICE="Ting-Ting"

# Output directory
OUT_DIR="log-viewer/static/audio"
mkdir -p "$OUT_DIR"

echo "Generating generic sounds..."
say -v "$VOICE" "碰" -o "$OUT_DIR/pon.m4a"
say -v "$VOICE" "杠" -o "$OUT_DIR/kan.m4a"
say -v "$VOICE" "胡" -o "$OUT_DIR/ron.m4a"
say -v "$VOICE" "自摸" -o "$OUT_DIR/tsumo.m4a"
say -v "$VOICE" "定缺" -o "$OUT_DIR/dingque.m4a"

echo "Generating tile sounds..."
# Wan (m)
say -v "$VOICE" "一万" -o "$OUT_DIR/1m.m4a"
say -v "$VOICE" "二万" -o "$OUT_DIR/2m.m4a"
say -v "$VOICE" "三万" -o "$OUT_DIR/3m.m4a"
say -v "$VOICE" "四万" -o "$OUT_DIR/4m.m4a"
say -v "$VOICE" "五万" -o "$OUT_DIR/5m.m4a"
say -v "$VOICE" "六万" -o "$OUT_DIR/6m.m4a"
say -v "$VOICE" "七万" -o "$OUT_DIR/7m.m4a"
say -v "$VOICE" "八万" -o "$OUT_DIR/8m.m4a"
say -v "$VOICE" "九万" -o "$OUT_DIR/9m.m4a"

# Pin (p) - Tong
say -v "$VOICE" "一筒" -o "$OUT_DIR/1p.m4a"
say -v "$VOICE" "二筒" -o "$OUT_DIR/2p.m4a"
say -v "$VOICE" "三筒" -o "$OUT_DIR/3p.m4a"
say -v "$VOICE" "四筒" -o "$OUT_DIR/4p.m4a"
say -v "$VOICE" "五筒" -o "$OUT_DIR/5p.m4a"
say -v "$VOICE" "六筒" -o "$OUT_DIR/6p.m4a"
say -v "$VOICE" "七筒" -o "$OUT_DIR/7p.m4a"
say -v "$VOICE" "八筒" -o "$OUT_DIR/8p.m4a"
say -v "$VOICE" "九筒" -o "$OUT_DIR/9p.m4a"

# Sou (s) - Tiao
say -v "$VOICE" "一条" -o "$OUT_DIR/1s.m4a"
say -v "$VOICE" "二条" -o "$OUT_DIR/2s.m4a"
say -v "$VOICE" "三条" -o "$OUT_DIR/3s.m4a"
say -v "$VOICE" "四条" -o "$OUT_DIR/4s.m4a"
say -v "$VOICE" "五条" -o "$OUT_DIR/5s.m4a"
say -v "$VOICE" "六条" -o "$OUT_DIR/6s.m4a"
say -v "$VOICE" "七条" -o "$OUT_DIR/7s.m4a"
say -v "$VOICE" "八条" -o "$OUT_DIR/8s.m4a"
say -v "$VOICE" "九条" -o "$OUT_DIR/9s.m4a"

echo "Done!"
