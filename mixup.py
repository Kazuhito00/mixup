import numpy as np


def mixup(image_batch, label_batch, alpha=0.2, is_debug=False):
    batch_size = len(image_batch)

    weights = np.random.beta(alpha, alpha, batch_size)

    index = np.random.permutation(batch_size)

    x1, x2 = image_batch, image_batch[index]
    x = np.array([
        x1[i] * weights[i] + x2[i] * (1 - weights[i])
        for i in range(len(weights))
    ])

    y1 = np.array(label_batch).astype(np.float)
    y2 = np.array(np.array(label_batch)[index]).astype(np.float)
    y = np.array([
        y1[i] * weights[i] + y2[i] * (1 - weights[i])
        for i in range(len(weights))
    ])

    if not is_debug:
        return x, y
    else:
        return x, y, x1, y1, x2, y2, weights
