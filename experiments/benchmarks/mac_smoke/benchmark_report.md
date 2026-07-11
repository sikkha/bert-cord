# Training benchmark — cpu / fp32

> Performance probe only. **Not** a scientific/convergence result; the config is not modified. Warmup steps are excluded from all rates.

## Throughput

- Steps/s: **3.032**
- Tokens/s: **692.307**
- Samples/s: 24.256
- Measured steps: 9 (warmup 3)

## Latency

- Median: 256.200 ms | p90: 503.561 ms | p99: 510.720 ms
- Forward: 123.342 ms | Backward: 162.245 ms | Optimizer: 43.861 ms | Dataloader wait: 0.368 ms

## Resources & config

- Peak RAM: 900.691 MB | Peak VRAM: n/a
- Effective batch: 16 (per-device 8 × accum 2) | seq len 32
- Total tokens processed: 2,743
- Eval overhead: 0.219 s | Checkpoint overhead: 12.194 s
- Params: 27,010,304

## Batch-size probe (diagnostic; config unchanged)

- Candidates: [8, 16, 32] @ seq 32
- Largest that fit: **32**

## Plots

![step_latency.png](plots/step_latency.png)
![phase_breakdown.png](plots/phase_breakdown.png)

