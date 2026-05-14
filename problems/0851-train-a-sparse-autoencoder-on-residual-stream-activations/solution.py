import numpy as np

def train_sae(X, W_enc, W_dec, b_enc, lr, l1_coef, num_steps):
    X = np.array(X, dtype=float)
    W_enc = np.array(W_enc, dtype=float)
    W_dec = np.array(W_dec, dtype=float)
    b_enc = np.array(b_enc, dtype=float)
    N, D = X.shape

    for _ in range(num_steps):
        pre   = X @ W_enc + b_enc
        z     = np.maximum(0, pre)
        x_hat = z @ W_dec

        d_xhat  = 2 * (x_hat - X) / (N * D)
        d_W_dec = z.T @ d_xhat
        d_z     = d_xhat @ W_dec.T + l1_coef / N
        d_pre   = d_z * (pre > 0)
        d_W_enc = X.T @ d_pre
        d_b_enc = np.sum(d_pre, axis=0)

        W_enc -= lr * d_W_enc
        W_dec -= lr * d_W_dec
        b_enc -= lr * d_b_enc

    pre   = X @ W_enc + b_enc
    z     = np.maximum(0, pre)
    x_hat = z @ W_dec
    return float(np.mean((x_hat - X)**2) + l1_coef * np.mean(np.sum(z, axis=1)))