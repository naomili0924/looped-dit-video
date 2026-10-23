STRIDE, F, K_MAX, K_SET = 2, 33, 17, (1, 5, 9, 17)


def n_windows(start, end):
    fut = (F - 1) * STRIDE
    a, n = min(start + K_MAX * STRIDE, end - 1 - fut), 0
    while a + fut <= end - 1:
        if not [k for k in K_SET if a - k * STRIDE >= start]:
            break
        n += 1
        a += F * STRIDE
    return n
