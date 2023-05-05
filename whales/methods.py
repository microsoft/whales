# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import numpy as np
import torch
from torch import Tensor
from torch.nn.modules import Conv2d, Module
from tqdm import tqdm


class LocalContextStandardization(Module):
    def __init__(self, in_channels: int = 3, kernel_size: int = 9, shift_val=None):
        super().__init__()

        self.shift_val = shift_val

        weights = torch.nn.Parameter(
            torch.zeros(
                in_channels, in_channels, kernel_size, kernel_size, dtype=torch.float32
            ),
            requires_grad=False,
        )
        for i in range(in_channels):
            weights[i, i] = (
                torch.ones(kernel_size, kernel_size, dtype=torch.float32)
                / kernel_size**2.0
            )

        self.conv = Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding="same",
            padding_mode="replicate",
            bias=False,
        )
        self.conv.weight = weights

    def forward(self, x: Tensor) -> Tensor:
        if self.shift_val is not None:
            x = x - self.shift_val
        else:
            x = x - x.mean(dim=(0, 2, 3), keepdim=True)
        mu = self.conv(x)
        squares = self.conv(x**2.0)
        variance = squares - mu**2.0
        return (x - mu) / (torch.sqrt(variance) + 1e-8)


def apply_rolling_standardization(data, device, patch_size, kernel_size):
    """A helper function for running a LocalContextStandardization model on an
    input that is too large to fit in GPU memory.
    """
    num_channels, height, width = data.shape

    half_kernel = kernel_size // 2
    shift_val = (
        torch.from_numpy(data.mean(axis=(1, 2), keepdims=True).astype(np.float32))
        .unsqueeze(0)
        .to(device)
    )

    model = LocalContextStandardization(
        num_channels, kernel_size=kernel_size, shift_val=shift_val
    ).to(device)

    y_options = list(range(0, height, patch_size - kernel_size))
    x_options = list(range(0, width, patch_size - kernel_size))
    num_y = len(y_options)
    num_x = len(x_options)

    output = np.zeros((num_channels, height, width), dtype=np.float32)
    for i, y in enumerate(tqdm(y_options)):
        for j, x in enumerate(x_options):
            p_input = (
                torch.from_numpy(
                    data[:, y : y + patch_size, x : x + patch_size].astype(np.float32)
                )
                .unsqueeze(0)
                .to(device)
            )
            p_output = model(p_input).cpu().numpy().squeeze()

            # This is all split up because I'm pretty sure that you need to slice differently for the edges of the input, but I haven't worked that out
            if i == 0:
                if j == 0:
                    output[:, y : y + patch_size, x : x + patch_size] = p_output
                elif j == num_x - 1:
                    output[:, y : y + patch_size, x : x + patch_size] = p_output
                else:
                    output[:, y : y + patch_size, x : x + patch_size] = p_output
            elif j == 0:
                if i == num_y - 1:
                    output[:, y : y + patch_size, x : x + patch_size] = p_output
                else:
                    output[:, y : y + patch_size, x : x + patch_size] = p_output
            elif i == num_y - 1:
                if j == num_x - 1:
                    output[:, y : y + patch_size, x : x + patch_size] = p_output
                else:
                    output[:, y : y + patch_size, x : x + patch_size] = p_output
            elif j == num_x - 1:
                output[:, y : y + patch_size, x : x + patch_size] = p_output
            else:
                output[
                    :,
                    y + half_kernel : y + patch_size - half_kernel,
                    x + half_kernel : x + patch_size - half_kernel,
                ] = p_output[:, half_kernel:-half_kernel, half_kernel:-half_kernel]

    return output


def apply_chunked_standardization(data, step_size=1024, nodata=None):
    num_channels, height, width = data.shape
    deviations = np.zeros((num_channels, height, width), dtype=np.float32)
    for y in tqdm(range(0, height, step_size)):
        for x in range(0, width, step_size):
            chunk = data[:, y : y + step_size, x : x + step_size]

            if nodata is not None:
                mask = chunk != nodata
                means = np.mean(chunk, axis=(1, 2), dtype=np.float64, keepdims=True, where=mask)
                stdevs = np.std(chunk, axis=(1, 2), dtype=np.float64, keepdims=True, where=mask)
            else:
                means = np.mean(chunk, axis=(1, 2), dtype=np.float64, keepdims=True)
                stdevs = np.std(chunk, axis=(1, 2), dtype=np.float64, keepdims=True)

            if np.all(stdevs == 0):
                continue

            deviations[:, y : y + step_size, x : x + step_size] = (
                chunk - means
            ) / stdevs

    return deviations


def apply_gmm(data, num_samples=10000, num_components=10):
    num_channels, height, width = data.shape
    r = 0

    x_all = np.zeros((num_samples, num_channels), dtype=np.float32)
    for i in range(num_samples):
        x = np.random.randint(r, data.shape[2] - r)
        y = np.random.randint(r, data.shape[1] - r)
        while np.all(data[:, y, x] == 0):
            x = np.random.randint(r, data.shape[2] - r)
            y = np.random.randint(r, data.shape[1] - r)
        x_all.append(np.abs(data[:, y, x]))
    x_all = np.array(x_all)


if __name__ == "__main__":
    test_input = np.random.randn(3, 1024 * 4, 1024 * 4)
    deviations = apply_rolling_standardization(
        test_input, torch.device("cuda:0"), 1024, 9
    )
