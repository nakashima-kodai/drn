import os


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def denorm(image):
    # [-1, 1] -> [0, 255]
    return (image + 1.0) / 2.0
