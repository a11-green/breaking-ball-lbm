# テクニカルノート（LaTeX）

本プロジェクトの理論と実装をまとめたテクニカルノート。[`a11-green/latex-docker-vscode`](https://github.com/a11-green/latex-docker-vscode) をテンプレートとして、このディレクトリ単体をビルド環境として持たせている（リポジトリ全体ではなく `docs/technical-note/` を開く）。

内容は `docs/design/DESIGN.md`（開発の過程で得られた検証結果・設計判断を随時記録している設計ノート）をもとに，理論と実装の要点を整理し直したものである。日々の詳細な記録はそちらを参照し，本ノートは理論的背景・実装方法・検証結果のまとまった記述を目的とする。

## ビルド方法

### Makefile（Docker、推奨・最速で試せる）

Docker が使える環境であれば、VSCodeを介さずこのディレクトリで直接：

```sh
cd docs/technical-note
make            # main.pdf をビルド
make clean      # 中間ファイルとPDFを削除
```

ローカルの \TeX Live を使う場合は `USE_DOCKER=no make`。

### VSCode Dev Container

1. VSCode で `docs/technical-note/` をワークスペースのルートとして開く（リポジトリ全体を開いた状態からは `.devcontainer` が拾われない）。
2. コマンドパレットから `Dev Containers: Reopen in Container` を実行する。
3. `main.tex` を開き、`LaTeX Workshop: Build LaTeX project`（レシピ `compile`）を実行する。

参考文献（`ref.bib`）を使うため、初回ビルドでは `upbibtex` が走る分だけ通常のビルドより1回多く `latexmk` の反復が必要になることがある——`make` / LaTeX Workshop のどちらも自動で処理するので操作は変わらない。

## 構成

```
docs/technical-note/
├─ main.tex                 文書の骨格（\input で各節を読み込む）と参考文献設定
├─ setting.tex              プリアンブル・自作コマンド（テンプレートのまま流用）
├─ ref.bib                  参考文献データベース（BibTeX）
├─ sections/                本文（節ごとに分割）
│   ├─ intro.tex                はじめに
│   ├─ aerodynamics.tex          野球ボールの空気力学的背景
│   ├─ lbm.tex                   格子ボルツマン法の基礎理論
│   ├─ collision.tex             衝突演算子と数値安定性（中心モーメント/キュムラント）
│   ├─ turbulence.tex            乱流モデリング（LES/Smagorinsky）
│   ├─ geometry_boundary.tex     縫い目形状のモデリングと移動境界条件
│   ├─ coupling.tex              密結合方式による軌道計算
│   ├─ gpu.tex                   GPU実装
│   ├─ validation.tex            検証（V&V-1〜5）
│   └─ conclusion.tex            まとめと今後の課題
├─ figure/                  図（`seam.svg`／`seam.pdf`：`scripts/plot_seam.jl` の出力をPDF化したもの）
└─ .devcontainer/, .vscode/, Makefile, .latexmkrc, mkdiff.sh 等
   テンプレートそのままの構成。詳しい説明はテンプレート側の README を参照。
```

節を追加・分割する場合は `sections/` に `.tex` ファイルを足し、`main.tex` に `\input{sections/ファイル名}` を1行追記すればよい。

`\author` は仮の値のままなので、内容を進める際に差し替えること。

## 図の追加について

このビルド環境は `uplatex` + `dvipdfmx` なので、`\includegraphics` に渡せるのは PDF・PNG・JPEG・BMP 等であり、SVGを直接読み込むことはできない。SVGで図を作った場合は、事前にPDFへ変換したものを `figure/` に置き、`.tex` 側ではそのPDFを参照する（`seam.svg`／`seam.pdf` の組がその実例）。
