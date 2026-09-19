# テクニカルノート（LaTeX）

本プロジェクトの理論と実装をまとめたテクニカルノート。[`a11-green/latex-docker-vscode`](https://github.com/a11-green/latex-docker-vscode) をテンプレートとして、このディレクトリ単体をビルド環境として持たせている（リポジトリ全体ではなく `docs/technical-note/` を開く）。

現時点では `main.tex` に「はじめに」節のみを記載している。まずはビルドが通ることの確認用で、以降のセクション（LBM理論、乱流モデリング、境界条件と軌道結合、GPU実装、V&V）は `docs/design/DESIGN.md` の内容をもとに順次追加していく。

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

## 構成

テンプレートそのままの構成（`.devcontainer/`、`.vscode/`、`Makefile`、`.latexmkrc`、`mkdiff.sh` など）に、`main.tex`（本文）と `setting.tex`（プリアンブル・自作コマンド、テンプレートのまま流用）を加えたもの。詳しい説明はテンプレート側の README を参照。

`\author` は仮の値のままなので、内容を進める際に差し替えること。
