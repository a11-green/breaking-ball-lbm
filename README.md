# breaking-ball-lbm

格子ボルツマン法（LBM）による変化球（curveball等）軌道シミュレーション。3次元乱流をGPUで並列計算し、球速・回転数・回転軸・縫い目向きなど様々な条件下でのボールの空気力学的挙動と軌道を解析する。

設計書・理論調査は [`docs/design/DESIGN.md`](docs/design/DESIGN.md) を参照。

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
julia --project=. -e 'using Pkg; Pkg.test()'
```

### GPU（要 NVIDIA GPU）

CUDA と StaticArrays は弱依存（`[weakdeps]`）なので、**プロジェクト環境ではなくデフォルト環境に**入れる。
`--project=.` を付けて `Pkg.add` すると `[deps]` と `[weakdeps]` が衝突してエラーになる。

```bash
julia -e 'using Pkg; Pkg.add(["CUDA", "StaticArrays"])'   # デフォルト環境へ
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
