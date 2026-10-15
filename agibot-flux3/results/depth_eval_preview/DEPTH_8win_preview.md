# Loop depth: background vs motion

Same noise at every depth. Moving region = pixels that move in the real clip (real prompt) or in the deepest output (alternative prompts). PSNR bg / moving: pixels outside / inside that region. Flow metrics as in MOTION.md. `motion` = mean compensated flow magnitude of the output inside the region (px/frame).

## Per depth, real prompt (mean over cases)

| model | k | N | n | PSNR bg | PSNR moving | LPIPS | dir cos | mag ratio | region cos | motion px/fr |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| loop2500 | 1 | 1 | 8 | 22.360 | 10.280 | 0.156 | 0.182 | 0.771 | 0.214 | 1.255 |
| loop2500 | 1 | 2 | 8 | 21.690 | 10.183 | 0.170 | 0.248 | 0.796 | 0.314 | 1.321 |
| loop2500 | 1 | 4 | 8 | 21.483 | 10.080 | 0.176 | 0.216 | 0.824 | 0.240 | 1.404 |
| loop2500 | 1 | 8 | 8 | 22.529 | 10.062 | 0.160 | 0.214 | 0.638 | 0.197 | 1.134 |
| loop2500 | 9 | 1 | 8 | 24.072 | 11.014 | 0.133 | 0.313 | 0.550 | 0.350 | 0.972 |
| loop2500 | 9 | 2 | 8 | 23.613 | 10.727 | 0.129 | 0.339 | 0.741 | 0.371 | 1.388 |
| loop2500 | 9 | 4 | 8 | 23.003 | 10.879 | 0.130 | 0.400 | 0.879 | 0.395 | 1.561 |
| loop2500 | 9 | 8 | 8 | 23.779 | 10.695 | 0.131 | 0.362 | 0.620 | 0.301 | 1.224 |

static baseline (copy the first kept frame): PSNR bg 32.571, PSNR moving 9.161

## Change between consecutive depths, real prompt

| model | k | transition | n | area in | delta in | delta out | concentration | mass in |
|---|--:|---|--:|--:|--:|--:|--:|--:|
| loop2500 | 1 | N1->N2 | 8 | 0.082 | 20.294 | 4.720 | 4.438 | 0.269 |
| loop2500 | 1 | N2->N4 | 8 | 0.082 | 17.341 | 3.808 | 4.806 | 0.274 |
| loop2500 | 1 | N4->N8 | 8 | 0.082 | 25.181 | 5.071 | 4.708 | 0.280 |
| loop2500 | 9 | N1->N2 | 8 | 0.082 | 25.284 | 5.306 | 5.084 | 0.286 |
| loop2500 | 9 | N2->N4 | 8 | 0.082 | 19.887 | 3.661 | 5.629 | 0.295 |
| loop2500 | 9 | N4->N8 | 8 | 0.082 | 28.161 | 4.302 | 7.005 | 0.354 |

concentration > 1: the extra loops change the moving region more (per pixel) than the background; mass in > area in: the moving region receives a larger share of the change than its area share.

## By case class, real prompt

| model | k | class | N | n | dir cos | mag ratio | PSNR moving |
|---|--:|---|--:|--:|--:|--:|--:|
| loop2500 | 1 | easy | 1 | 4 | 0.340 | 0.824 | 10.987 |
| loop2500 | 1 | easy | 2 | 4 | 0.339 | 0.925 | 11.008 |
| loop2500 | 1 | easy | 4 | 4 | 0.286 | 1.014 | 10.894 |
| loop2500 | 1 | easy | 8 | 4 | 0.145 | 0.747 | 10.037 |
| loop2500 | 1 | hard | 1 | 3 | 0.017 | 0.580 | 9.226 |
| loop2500 | 1 | hard | 2 | 3 | 0.161 | 0.486 | 8.959 |
| loop2500 | 1 | hard | 4 | 3 | 0.208 | 0.498 | 8.866 |
| loop2500 | 1 | hard | 8 | 3 | 0.138 | 0.569 | 8.945 |
| loop2500 | 1 | static | 1 | 1 | 0.047 | 1.134 | 10.616 |
| loop2500 | 1 | static | 2 | 1 | 0.147 | 1.214 | 10.557 |
| loop2500 | 1 | static | 4 | 1 | -0.044 | 1.045 | 10.464 |
| loop2500 | 1 | static | 8 | 1 | 0.716 | 0.414 | 13.513 |
| loop2500 | 9 | easy | 1 | 4 | 0.560 | 0.821 | 11.566 |
| loop2500 | 9 | easy | 2 | 4 | 0.604 | 0.732 | 11.412 |
| loop2500 | 9 | easy | 4 | 4 | 0.730 | 0.882 | 11.814 |
| loop2500 | 9 | easy | 8 | 4 | 0.393 | 0.752 | 10.563 |
| loop2500 | 9 | hard | 1 | 3 | 0.050 | 0.342 | 9.730 |
| loop2500 | 9 | hard | 2 | 3 | 0.157 | 0.607 | 9.819 |
| loop2500 | 9 | hard | 4 | 3 | 0.141 | 0.638 | 9.759 |
| loop2500 | 9 | hard | 8 | 3 | 0.274 | 0.627 | 9.978 |
| loop2500 | 9 | static | 1 | 1 | 0.111 | 0.092 | 12.653 |
| loop2500 | 9 | static | 2 | 1 | -0.173 | 1.178 | 10.711 |
| loop2500 | 9 | static | 4 | 1 | -0.148 | 1.586 | 10.495 |
| loop2500 | 9 | static | 8 | 1 | 0.500 | 0.068 | 13.377 |