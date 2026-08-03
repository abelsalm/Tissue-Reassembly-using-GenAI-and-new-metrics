# Hybrid MMD benchmark decision

Environment: PyTorch 2.8.0+cu128, NVIDIA H200. Measurements used CUDA events
after warm-up. The GPU was shared, so absolute times may vary; dispatch choices
were based on repeated relative measurements and numerical tests.

## Observed bottlenecks and gains

- Direct pair-distance sampling removed the full `n×n` distance construction.
  At `n=512`, its forward+backward median was about 1.7 ms versus 6.2 ms for
  the baseline. At `n=1000`, it was about 1.9 ms versus 8.9 ms.
- The vectorized all-band GT cache reduced repeated distance/PCA work. At
  `n=1000`, repeated measurements were roughly 3–5x faster than the baseline,
  at the cost of a higher transient peak because all sigma bands are resident
  together.
- A single tiled exact class can be slower than `cdist`, but batching matters:
  eight exact classes of 512 cells took 4.34 ms together versus 9.81 ms when
  the baseline local MMD was called eight times.
- The sampled whole-slice path used about 79 MiB in the micro-benchmark and
  remained around 2–3 ms from 512 to 5000 cells. The baseline grows
  quadratically and was deliberately not run at 5000 in the shared session.
- At `n=5000`, the final stratified sampled local term took about 7.5 ms and
  the vectorized GT cache about 20.5 ms, with a 5.1 GiB transient peak during all-band frame
  construction.

## Accuracy-driven dispatch

Global iid pair sampling gave noisy per-cell gradients for medium classes,
even when its scalar loss was close. Stratifying by prediction source improves
coverage, but the cancellation intrinsic to MMD still makes exact gradients
preferable for the common MERFISH class sizes.

The selected policy is therefore:

- exact, padded-bucket, tiled local MMD through 1024 cells;
- stratified incomplete U-statistics only above 1024 cells;
- sampled whole-slice MMD at the configured 131072-pair budget;
- sampled pair-distance-distribution MMD with direct cell-pair draws;
- exact nearest-GT support for classes through 1024 cells, and sampled
  prediction queries (each against all GT cells) above that threshold.

This keeps the approximately 570–1000-cell dominant MERFISH_ABC classes exact
while removing the most expensive redundant and whole-slide quadratic passes.

## Triton decision

A custom ragged Triton kernel is not currently justified. The PyTorch bucket
path already batches the common exact classes and was about 2.3x faster than
the per-class loop in the representative eight-class benchmark. After direct
pair sampling, the pair-distance and whole-slice terms dominate much less.

The benchmark script retains the per-component measurements needed to revisit
this decision:

```bash
python -m metrics.benchmark_train_mmds_hybrid \
  --sizes 256 512 730 850 1000 5000 --repeats 5 --warmup 2
```

Triton should be reconsidered only if a full-step profiler shows the exact
bucket kernel dominating epoch time after compilation warm-up.
