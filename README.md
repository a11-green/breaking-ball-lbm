# breaking-ball-lbm

格子ボルツマン法（LBM）による変化球（curveball等）軌道シミュレーション。3次元乱流をGPUで並列計算し、球速・回転数・回転軸・縫い目向きなど様々な条件下でのボールの空気力学的挙動と軌道を解析する。

設計書・理論調査は [`docs/design/DESIGN.md`](docs/design/DESIGN.md) を参照。

## 現在の状態

ロードマップP1（CPU参照実装）まで完了。D3Q19 + BGK、周期境界のソルバーが動作し、減衰Taylor-Green渦の解析解に対して2次精度の収束を確認済み（V&V-1）。

## 動かし方

Julia 1.10以降が必要。

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

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
