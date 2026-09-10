# Measurements

Raw numbers behind the claims in the README. All on an RTX 2060 6 GB (sm_75)
under WSL2 Ubuntu 22.04, 8 GB RAM cap, 4 cores, torch 2.6.0+cu124, gsplat 1.5.3.

Three clips, all one shot stock footage never intended for reconstruction:

| id | content | source |
|---|---|---|
| hotel | 21 s indoor, camera dollies past a bed | 1920x1080 24 fps |
| house | 13 s indoor, camera dollies down a narrow corridor | 1920x1080 24 fps |
| town | 23 s near nadir aerial over a village on water | 1920x1080 25 fps |

## Reconstruction pixel budget

Largest chunk that fits in one forward pass. The binding constraint is total
pixels, not view count, at roughly 700k.

| views x resolution | total px | peak VRAM | result |
|---|---|---|---|
| 4 x 294x518 | 609k | 4.75 GB | ok |
| 6 x 252x448 | 677k | 5.27 GB | ok |
| 10 x 182x336 | 611k | 4.78 GB | ok |
| 24 x 126x224 | 677k | 5.30 GB | ok |
| 12 x 182x336 | 733k | | OOM |
| 5 x 294x518 | 762k | | OOM |

Predicted correctly on three held out configurations.

## Weight precision

| dtype | weights on GPU | free | result |
|---|---|---|---|
| fp32 | 4.91 GB | 0.42 GB | cannot run |
| fp16 | 2.46 GB | 2.88 GB | 456876 of 456876 ray values NaN |
| bf16 | 2.46 GB | 2.88 GB | works |

## Chunk overlap

Hotel clip, 37 frames. Fewer chunk transitions means less accumulated drift,
so lower overlap is both faster and tighter.

| overlap | chunks | inference | stitch mean | stitch max |
|---|---|---|---|---|
| 3 | 34 | 158.9 s | 6.3 mm | 21.5 mm |
| 2 | 18 | 86.4 s | 4.9 mm | 18.3 mm |

## Reconstruction timing

Hotel clip, 37 frames, overlap 2.

| stage | time | share |
|---|---|---|
| probe | 0.7 s | 0.4% |
| frame extraction | 1.0 s | 0.5% |
| sharpness filter | 1.0 s | 0.5% |
| model load | 39.6 s | 19.2% |
| inference | 86.4 s | 64.2% |
| output | 5.7 s | 2.8% |

Scaling: about 40 s fixed plus 2.34 s per frame.

One run measured 3x slower across every stage. The GPU was power capped
(`SW_POWER_CAP`, 154 of 160 W, 1755 of 2100 MHz) at 73 C after hours of
training, with the 4.6 GB checkpoint evicted from page cache. A later run on a
settled machine returned to baseline. Do not benchmark immediately after a
long training run.

## Capture quality versus reconstruction quality

`fwd frac` is the fraction of camera motion along the optical axis.
`perp B/D` is the perpendicular baseline to depth ratio.

| clip | raw B/D | fwd frac | perp B/D | reprojection corr |
|---|---|---|---|---|
| hotel | 0.0395 | 0.82 | 0.0226 | 0.758 |
| house | 0.0619 | 0.99 | 0.0087 | 0.424 |
| town | 0.0197 | 0.29 | 0.0189 | 0.685 |

Raw B/D versus quality: Spearman -0.50 (wrong direction).
Perpendicular B/D versus quality: Spearman +1.00.

n=3, so this is suggestive rather than established, but the mechanism is
standard: perpendicular motion is what produces parallax.

Note house has the best stitch residual of the three (5.4 mm) and the worst
geometry. Stitch residual measures chunk agreement, not correctness.

## Splat training

Hotel clip. All runs 10000 iterations unless noted.

| config | l1 | Gaussians | peak VRAM | time |
|---|---|---|---|---|
| MCMC strategy, cap 300k | 0.0282 | 953k frozen | 1.66 GB | 39 min |
| MCMC strategy, cap 2.5M | diverged | exploded | 5.34 GB | OOM at 2400 |
| Default strategy, 1554 px, dense init | 0.0117 | 525k | 2.37 GB | 39.3 min |
| Default strategy, 1554 px, sparse init | 0.0121 | 318k | 0.82 GB | 16.2 min |
| Default strategy, 1890 px, sparse init | 0.0120 | 297k | 0.96 GB | 18.3 min |
| Default strategy, 1890 px, grow-grad 5e-5 | **0.0098** | **1.51M** | 2.79 GB | 42.2 min |

Matched resolution comparison of the last two, same view and iteration:

| metric | 297k Gaussians | 1.5M Gaussians |
|---|---|---|
| mean L1 | 0.0124 | 0.0098 |
| L1 on top 10% gradient pixels | 0.0333 | 0.0273 |
| L1 on top 3% gradient pixels | 0.0597 | 0.0490 |
| detail energy reproduced (1.0 ideal) | 0.8285 | 0.8937 |

### Rasteriser memory is superlinear

| Gaussians | peak VRAM |
|---|---|
| 1.98M | 3.55 GB |
| 2.19M | 3.91 GB |
| 2.29M | 4.35 GB |
| 2.41M | 5.34 GB then OOM |

A 5% increase in count between the last two rows cost 23% more memory.

## Scene difficulty

| clip | perp B/D | detail energy reproduced |
|---|---|---|
| hotel | 0.0226 | 0.894 |
| town | 0.0189 | 0.599 |

A 16% difference in perpendicular parallax produced a 33% difference in
reproduced detail, larger than any training setting moved.

## Nadir aerial scale drift

Town clip. A drone flies smoothly and water is flat, so both of these should
be near zero.

| measurement | value |
|---|---|
| estimated altitude wobble per frame | 0.100 m |
| actual lateral travel per frame | 0.104 m |
| wobble to travel ratio | 0.96 (smooth flight would be under 0.1) |
| water surface plane fit residual | 0.405 m std, 0.766 m p95 |
| scene lateral extent | 13.7 m |

Estimated altitude jitters by as much as the drone actually moves. This is the
depth scale ambiguity: with no horizon in frame there is little to distinguish
a higher camera over a farther scene from a lower camera over a nearer one.
