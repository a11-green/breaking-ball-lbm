# breaking-ball-lbm

格子ボルツマン法（LBM）による変化球（curveball等）軌道シミュレーション。3次元乱流をGPUで並列計算し、球速・回転数・回転軸・縫い目向きなど様々な条件下でのボールの空気力学的挙動と軌道を解析する。

設計書・理論調査は [`docs/design/DESIGN.md`](docs/design/DESIGN.md) を参照。理論と実装のテクニカルノート（LaTeX、執筆中）は [`docs/technical-note/`](docs/technical-note/)。

## 現在の状態

CPU側は一通り動作し、解析解に対する検証を通っている。

- **格子**: D3Q19（検証用）/ D3Q27（本番用）
- **衝突演算子**: BGK と 中心モーメント（キュムラント緩和）。後者はせん断層で τ=0.5001（Re≈19万）まで安定
- **境界条件**: 補間バウンスバック（曲面・回転壁）、運動量交換法による力とトルク
- **乱流モデル**: Smagorinsky
- **ジオメトリ**: 縫い目のパラメトリック生成と符号付き距離場
- **GPU**: AAパターンのカーネルをRTX 3060 Tiで検証済み。FP32で **2,558 MLUPS**（実効552 GB/s、メモリ帯域に張り付いている）。壁面境界のGPU化は未実装

## 動かし方

Julia 1.10以降が必要。

```bash
julia --project=. -e 'using Pkg; Pkg.test()'    # POSIX shells, PowerShell
julia --project=. -e "using Pkg; Pkg.test()"    # cmd.exe (単一引用符は文字列にならない)
```

### GPU（要 NVIDIA GPU）

CUDA と StaticArrays は弱依存（`[weakdeps]`）なので、**プロジェクト環境ではなくデフォルト環境に**入れる。
`--project=.` を付けて `Pkg.add` すると `[deps]` と `[weakdeps]` が衝突してエラーになる。

```bash
julia -e 'using Pkg; Pkg.add(["CUDA", "StaticArrays"])'         # POSIX shells, PowerShell
julia -e "using Pkg; Pkg.add([\"CUDA\", \"StaticArrays\"])"     # cmd.exe
julia --project=. scripts/benchmark_gpu.jl
```

デフォルト環境は常に `LOAD_PATH` に含まれるため、`--project=.` で実行してもこの2つは読み込まれ、拡張が有効になる。

このスクリプトはタイミングを測る前に、GPUの結果がCPU参照実装と一致するか、Taylor-Green渦の減衰から粘性が正しく再現されるかを自己検証する。検証に失敗した場合はタイミングを出さずに停止する。

```julia
using BreakingBallLBM
const BBL = BreakingBallLBM

τ  = 0.8
tg = TaylorGreen(u0 = 0.02, n = 64, ν = viscosity_from_tau(τ))
s  = LBMState(64, 64, 4, τ)
BBL.init!(s, tg)
run!(s, 500)
BBL.l2_velocity_error(s, tg, 500)   # (相対L2誤差, 絶対L2誤差)
```

## `scripts/run_pitch.jl` のオプション

投球1球を通しで計算するメインのドライバ。位置と空気力学量を並べた軌跡CSVと、要約をstdoutに出す。

```bash
julia --project=. scripts/run_pitch.jl --smoke              # 疎通確認（低解像度・短距離、1分程度）
julia --project=. scripts/run_pitch.jl --resolution 40 --domain 8
julia --project=. scripts/run_pitch.jl --config run.toml    # 設定ファイルから読む
julia --project=. scripts/run_pitch.jl --help                # 実行時の一覧（このセクションと同じ内容）
```

座標系は x が本塁方向、z が鉛直上向き、y が投手から見て左。

### 疎通確認

| オプション | 意味 |
|---|---|
| `--smoke` | 低解像度（12点/D）・短距離（0.4 m）に切り替え、他の全設定より前に処理される。配管の確認用で、出る係数は空力的な意味を持たない |
| `--cpu` | GPUがあってもホスト経路を強制する |

### 格子と解像度

| オプション | 既定値 | 意味 |
|---|---|---|
| `--resolution N` | 40 | ボール1直径あたりの格子点数。コストは概ね $N^4$（空間3乗×時間刻み）で効く |
| `--domain L` | 8.0 | 箱の一辺、直径単位。`--open` 併用時は横幅のみを決める（流れ方向は `--upstream`/`--downstream`） |
| `--precision f32\|f64` | f32 | 格子の数値精度。本番はf32（GPUのメモリ帯域で律速するため） |

### リファインメント（§6.5.1）

| オプション | 既定値 | 意味 |
|---|---|---|
| `--refine F[,G,...]` | なし（一様格子） | ボール周囲 F 直径の箱を2倍解像度に、その中の G 直径をさらに2倍に、と入れ子にする。8直径の箱で縫い目を意味のある解像度に持ち込む唯一の方法。最初の値は3以上を推奨、後続はそれより小さく |
| `--refine-wake W[,X,...]` | 0（立方体） | 各パッチを下流側にだけ W 直径延長する。剥離とせん断層は前方の滑らかな流れより長く伸びるため |

### 開放境界（§4.4.2.1）

| オプション | 既定値 | 意味 |
|---|---|---|
| `--open` | 無効（周期境界） | 流れ方向を流入・流出境界にする。周期境界ではボールが自分自身の後流の中を飛ぶことになり、V&V-2で最大の誤差要因と判明した（球に届く流れが指定値の3分の1しかなかった） |
| `--upstream D` | 3.0 | ボール前方の助走区間、直径単位 |
| `--downstream D` | 8.0 | ボール後方の後流区間、直径単位 |

### 投球条件

| オプション | 既定値 | 意味 |
|---|---|---|
| `--speed V` | 38.99 | リリース速度 (m/s)。既定値はWBC2023決勝・大谷翔平のスウィーパー（§8 V&V-5） |
| `--rpm R` | 2708.0 | 回転数 |
| `--axis x,y,z` | 0,0,1 | スピン軸（既定はサイドスピン） |
| `--distance D` | 18.44（本塁までの距離） | リリースから計測点までの距離 (m) |

### スピンアップ

| オプション | 既定値 | 意味 |
|---|---|---|
| `--spinup N` | 0（下の設定から自動計算） | 軌道を固定したまま流れを発達させるサブサイクル数 |
| `--spinup-flowthroughs F` | 2.0 | 上と同じことを箱のflow-through時間の倍数で指定する。後流が自分の長さに達するまでの目安。リファインメントを使う場合、最深レベルの境界層が定常に達するまで余分にかかることがあるため、3.0程度を推奨 |

### 物理モデル

| オプション | 既定値 | 意味 |
|---|---|---|
| `--smagorinsky C` | 0.1 | サブグリッド定数。0にすると衝突演算子自身の数値散逸のみに頼る |

### 出力

| オプション | 既定値 | 意味 |
|---|---|---|
| `--out FILE` | pitch.csv | 軌跡CSV。位置・速度・流入条件（物理値・格子値）・姿勢クォータニオン・力とトルク・体積力・実測平均流速・再カット数・残差など約40列 |
| `--snapshot N` | 0（出力しない） | Nサブサイクルごとに流れ場のスナップショットを書く。1フレームにつき2ファイル（流れ場のボリュームと、縫い目曲線＋スピン軸）＋それぞれの `.vtk.series` |
| `--snapshot-crop D` | 2.0 | 書き出す範囲の半幅、直径単位（ボール追従系の中心から） |
| `--snapshot-stride S` | 1 | S点おきに間引いて書く（勾配は間引く前の格子で計算済み） |
| `--snapshot-dir DIR` | snapshots | 出力先ディレクトリ。実行のたびに、このスクリプトが書いた形式のファイルだけを消してから書き直す |
| `--overview` | 無効 | スピンアップ直後に領域全体（格子・流れ場、各リファインメントレベルを別ファイルで）を1枚だけ書く |
| `--overview-stride S` | 4 | 上の間引き（最粗レベルを間引かずに書くと1GB近くなるため） |

### 設定ファイル

| オプション | 意味 |
|---|---|
| `--config FILE` | TOMLファイルから設定を読む。コマンドライン引数はそれより後に処理されるので上書きできる（`--config base.toml --resolution 50` は base.toml の resolution だけを差し替える）。未知のキーはエラーになる |

実行するたびに、解決済みの全設定が起動ログとTOMLとして `<out>-config.toml` に書き出される。ログをそのまま `--config` に渡せば同じ実行を再現できるので、**結果のログをそのまま渡すだけで実行条件も一緒に伝わる**。

`release`（リリース位置）・`recut_drift`（再カットの閾値）・`operator`（衝突演算子）・`rule`（バウンスバック規則）・`report_every`（stdout出力の間隔）・`max_cycles`（安全装置としての上限サブサイクル数）はコマンドラインからは変更できず、`--config` のTOMLファイルでのみ指定できる。

## `scripts/analyze_pitch.jl` — 結果の可視化・分析

`run_pitch.jl` が書いた軌跡CSVを読み、位置・流入条件・速度・スピン・力係数・ソルバの健全性（残差・再カット数）をまとめて図にする（§7.4、段階2）。CSVの列は姿勢クォータニオン（`qw,qx,qy,qz`）とスピン（`wx,wy,wz`）を持つので、3D軌道パネルには**そのCFD計算が実際に持っていた向きの縫い目**を、任意の時刻にスクラブして表示できる（`view_pitch.jl` の解析モデル版と同じ仕組み）。

捕手視点パネルには、`scripts/measure_break.jl`/`scripts/plot_break.jl` と同じpfx/induced/totalの3つの基準点を、実際の到達点への点線付きでマーカー表示する（`vx,vy,vz`列が無い古いCSVでは自動的に省略される）。

```bash
julia --project=. scripts/analyze_pitch.jl pitch.csv
julia --project=. scripts/analyze_pitch.jl pitch.csv --backend cairomakie --record report.png
julia --project=. scripts/analyze_pitch.jl pitch.csv --smooth 41
julia --project=. scripts/analyze_pitch.jl pitch.csv --reference "4-seam fastball"
julia --project=. scripts/analyze_pitch.jl pitch.csv --compare other_pitch.csv
julia --project=. scripts/analyze_pitch.jl --help
```

Makieのインストールは `view_pitch.jl` と同様（デフォルト環境に `GLMakie`/`WGLMakie`/`CairoMakie` のいずれかを入れる）。

| オプション | 既定値 | 意味 |
|---|---|---|
| `CSV`（位置引数）/ `--csv FILE` | pitch.csv | 読み込む軌跡CSV |
| `--smooth N` | 行数から自動選択 | 力係数パネルの移動平均の窓幅（サブサイクル数）。瞬時の運動量交換力は乱流のゆらぎで大きく振れるため、生の系列を薄く重ねた上に移動平均を太線で描く |
| `--seam-samples N` | 200 | 3Dパネルで縫い目曲線を描く点数 |
| `--ball-scale S` | 15.0 | 3Dパネルでのボールの拡大率（起動時の初期値。表示中も "ball ×" スライダーで変更できる） |
| `--compare FILE` | なし（繰り返し指定可） | 別の実行の軌跡CSVを重ね描きする（自分で計算した別の球種・別解像度との比較用） |
| `--reference NAME` | なし | `view_pitch.jl` の解析モデル（`PITCH_TYPES`）から汎用的な参考球種を重ね描きする。名前は `4-seam fastball`, `2-seam / sinker`, `sweeper`, `gyroball`, `12-6 curve`。**特定の投手の実測値ではなく典型的なパラメータ**（大谷投手個人のフォーシームの実測値をBaseball Savantで確認できていないため。§11参照） |
| `--reference-speed V` / `--reference-rpm R` | 参考球種自身の値 | `--reference` の球速・回転数を上書き |
| `--shift-y M` | 0.0 | 表示専用でCFD軌道のyだけをM平行移動する（再計算はしない）。実際の投手のリリースポイントの横位置に合わせて見せたい時用——変化量の数値には影響しない（3定義ともリリース位置からの相対値のため）。`--compare`・`--reference`はシフトされない |
| `--fps N` | 30.0 | 自動再生のフレームレート |
| `--backend` / `--web` / `--record` | `view_pitch.jl` と同じ | 描画バックエンドと出力先 |

読み込みは列名ベースで、必要な列（`t,x,y,z,speed,CD,CL,Cside,rpm,u_in_*,q*,w*,recuts,residual`）が無いCSVは、どの列が足りないかを明示するエラーで止まる——列が増減しても、足りない列だけを名指しできるようにするための設計（`src/postprocess/pitch_csv.jl`、`read_pitch_csv`/`require_columns`）。

**表示中の操作**（`--record` を使わない場合）: スクラブ用スライダーのほか、Reset/Play で自動再生（"speed (fps)" スライダーで再生中も速度を変更できる）、方位角(azimuth)・仰角(elevation)スライダーで3Dカメラを向ける、「MLB view」ボタンで放送中継風のアングル（投手の背後から本塁方向、やや見下ろし）に飛ぶ——ただしこのボタンの角度は未検証（このサンドボックスにMakieを入れられずGUIで確認できていない）ので、実際の中継映像と見比べてスライダーで追い込み、ちょうどいい値を教えてもらえれば既定値として焼き込む。

## `scripts/measure_break.jl` — CFD軌道からのpfx/induced/total変化量

`run_pitch.jl` の実行末尾に出る「Break測定は…re-integrate from the CSV with the Magnus component removed」の実装（§8 V&V-5）。解析係数モデルなら `no_magnus(aero)` がCL_slopeを0にするだけで済むが、CFDには「マグヌスを切った以外は同じ」実行が存在しない。代わりに、この実行自身が測定したC_D(t)（一定値ではなく実際の乱流変動込みの抗力履歴）から揚力・横力を完全に0とした抗力のみのモデルを作り、分岐点（リリース／プレート手前40ft）から再積分する。計算本体は `src/postprocess/measured_break.jl`（`measure_break`関数、Makie不要・`test/test_measured_break.jl`でテスト済み）にあり、このスクリプトは結果を表示するだけの薄いラッパー。

```bash
julia --project=. scripts/measure_break.jl pitch.csv
julia --project=. scripts/measure_break.jl pitch.csv --dt 1e-5
julia --project=. scripts/measure_break.jl --help
```

Makie不要・GPU不要の純粋な数値後処理。

## `scripts/plot_break.jl` — 変化量の比較グラフ

`measure_break.jl` と同じ数値（この実行のpfx/induced/total）を、`pitch_metrics`（`PITCH_TYPES`の解析係数モデル、既定は`sweeper`——WBC決勝ケースと同じ諸元）および報道された変化量（17インチ・32インチ、出典はDESIGN.md §8参照）と並べて棒グラフにする。

```bash
julia --project=. scripts/plot_break.jl pitch.csv
julia --project=. scripts/plot_break.jl pitch.csv --assume pfx
julia --project=. scripts/plot_break.jl pitch.csv --backend cairomakie --record break.png
julia --project=. scripts/plot_break.jl --help
```

| オプション | 既定値 | 意味 |
|---|---|---|
| `CSV`（位置引数）/ `--csv FILE` | pitch.csv | 読み込む軌跡CSV |
| `--dt T` | 1e-4 | 参照軌道の積分ステップ |
| `--reference NAME` | sweeper | 比較する解析係数モデルの球種（`PITCH_TYPES`） |
| `--assume DEF` | total | 報道値（17"/32"）の破線ラベルに使う定義名（pfx/induced/total）。**出典に定義の明記はなく**、ラベルの文言が変わるだけで計算には影響しない |
| `--backend` / `--web` / `--record` | `analyze_pitch.jl` と同じ | 描画バックエンドと出力先 |

報道値の破線は特定の棒に対応付けず、グラフ全体を横切る一本の参照線として描く——どの定義と比較して読むかは見る側に委ねる（DESIGN.md §11: 縦の変化量「32インチ」は3定義のいずれとも一致しないという未解決事項があるため）。
