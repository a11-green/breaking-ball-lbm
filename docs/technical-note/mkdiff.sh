#!/bin/bash

# ファイル名
TEX_FILE="main.tex"  # メインのtexファイル名に変更してください
OUTPUT_DIFF="diff.tex"  # 差分出力用のファイル名
PDF_OUTPUT="diff.pdf"  # 差分PDFの出力ファイル名

# 1. 前回のコミットをarchiveとして取得
mkdir -p archive  # 一時ディレクトリを作成
# git archive -o archive.tar HEAD~1  # 前回のコミットを取得
git show HEAD:$TEX_FILE > archive/$TEX_FILE
# tar -xf archive.tar -C archive  # 解凍

# 2. latexdiffで差分生成
latexdiff archive/$TEX_FILE $TEX_FILE > $OUTPUT_DIFF

# # 3. LaTeXをコンパイルして差分PDFを生成
# pdflatex $OUTPUT_DIFF
# bibtex $OUTPUT_DIFF  # 参考文献があれば
# pdflatex $OUTPUT_DIFF
# pdflatex $OUTPUT_DIFF  # 必要に応じて再コンパイル

# 4. 一時ファイルの削除
rm -rf archive archive.tar $OUTPUT_DIFF.{aux,log,blg,bbl}  # 不要なファイルを削除
