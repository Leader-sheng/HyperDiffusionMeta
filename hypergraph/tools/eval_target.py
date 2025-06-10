import numpy as np
from PIL import Image
from scipy.ndimage import zoom

def read_gif_to_numpy(gif_path):
    gif = Image.open(gif_path)
    frames = []

    for frame in range(gif.n_frames):
        gif.seek(frame)
        img = gif.convert("L")
        img_array = np.array(img)
        frames.append(img_array)

    return np.array(frames)


# 2. 4x4平均池化
def average_pooling(input_array, pool_size=4):
    # 计算新的尺寸
    new_shape = (input_array.shape[0], input_array.shape[1] // pool_size, input_array.shape[2] // pool_size)
    pooled_array = np.zeros(new_shape)

    for i in range(input_array.shape[0]):
        for j in range(0, input_array.shape[1], pool_size):
            for k in range(0, input_array.shape[2], pool_size):
                pooled_array[i, j // pool_size, k // pool_size] = np.mean(
                    input_array[i, j:j + pool_size, k:k + pool_size])

    return pooled_array


def compute_mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

input_gif_path = ""

reference_gif_path = ""

input_numpy = read_gif_to_numpy(input_gif_path)
reference_numpy = read_gif_to_numpy(reference_gif_path)

input_resized = average_pooling(input_numpy, pool_size=4)

print(input_resized.shape)
mse_value = compute_mse(input_resized, reference_numpy)
print(f"Mean Squared Error (MSE): {mse_value}")
