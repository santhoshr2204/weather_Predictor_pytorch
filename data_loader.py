import torch

def load_data():
    X = torch.tensor([
        [30, 70, 10],
        [28, 65, 12],
        [25, 80, 5],
        [22, 90, 4],
        [31, 50, 7]
    ], dtype=torch.float32)

    y = torch.tensor([
        [0],
        [0],
        [1],
        [1],
        [1]
    ], dtype=torch.float32)

    return X, y
