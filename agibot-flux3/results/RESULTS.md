# Results

Fixed windows per split (seeded), Euler 30 steps, CFG 4, shift 5; metrics against the
ground-truth future frames (PSNR/SSIM higher is better, LPIPS/FVD lower); judge = yes-rate with the true
caption minus yes-rate with a swapped caption (Qwen3-VL).

## test

| model | k | PSNR | SSIM | LPIPS | FVD (n) | judge adherence | n_recon |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0 | 9.09 | 0.374 | 0.765 | - | - | 100 |
| base | 1 | 17.03 | 0.664 | 0.288 | 407.6 (300) | 0.380 | 100 |
| base | 5 | 19.76 | 0.778 | 0.185 | - | - | 100 |
| base | 9 | 20.16 | 0.799 | 0.168 | - | - | 100 |
| ft_step6000 | 0 | 10.63 | 0.429 | 0.689 | - | - | 100 |
| ft_step6000 | 1 | 19.83 | 0.805 | 0.166 | 166.9 (300) | 0.460 | 100 |
| ft_step6000 | 5 | 21.63 | 0.840 | 0.129 | - | - | 100 |
| ft_step6000 | 9 | 21.77 | 0.848 | 0.123 | - | - | 100 |
