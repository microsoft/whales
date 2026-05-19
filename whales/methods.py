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
        self.min_stdev = 5  # A realistic minimum stdev for 0-2000 scaled TOA Maxar imagery

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
        variance = torch.clamp(squares - mu**2.0, min=0.0) # Calculate raw variance and clamp to 0.0 to prevent floating-point NaNs
        stdev = torch.sqrt(variance)
        stdev = torch.clamp(stdev, min=self.min_stdev)  # Clamp the standard deviation to a realistic physical baseline

        return (x - mu) / stdev


def apply_rolling_standardization(data, device, patch_size, kernel_size, nodata=None):
    """A helper function for running a LocalContextStandardization model on an
    input that is too large to fit in GPU memory.
    """
    num_channels, height, width = data.shape
    half_kernel = kernel_size // 2

    if nodata is not None:
        valid_mask = data != nodata
        mean_val = data.mean(axis=(1, 2), keepdims=True, where=valid_mask).astype(
            np.float32
        )
        clean_data = np.where(~valid_mask, mean_val, data)

    else:
        mean_val = data.mean(axis=(1, 2), keepdims=True).astype(np.float32)
        clean_data = data

    shift_val = (
        torch.from_numpy(mean_val.astype(np.float32))
        .unsqueeze(0)
        .to(device)
    )

    model = LocalContextStandardization(
        num_channels, kernel_size=kernel_size, shift_val=shift_val
    ).to(device)

    y_options = list(range(0, height, patch_size - kernel_size))
    x_options = list(range(0, width, patch_size - kernel_size))

    output = np.zeros((num_channels, height, width), dtype=np.float32)
    for i, y in enumerate(tqdm(y_options)):
        for j, x in enumerate(x_options):
            p_input = (
                torch.from_numpy(
                    clean_data[:, y : y + patch_size, x : x + patch_size].astype(np.float32)
                )
                .unsqueeze(0)
                .to(device)
            )
            p_output = model(p_input).cpu().numpy().squeeze(axis=0)

            # Edge-slicing logic
            y1_in, y2_in = y, min(y + patch_size, height)
            x1_in, x2_in = x, min(x + patch_size, width)
            # Define output coordinates (where to write in the giant array)
            # Crop the half_kernel unless we are touching the absolute image boundary
            y1_out = y1_in if y1_in == 0 else y1_in + half_kernel
            y2_out = y2_in if y2_in == height else y2_in - half_kernel
            x1_out = x1_in if x1_in == 0 else x1_in + half_kernel
            x2_out = x2_in if x2_in == width else x2_in - half_kernel

            # Define read coordinates (what part of the patch to keep)
            y1_read = 0 if y1_in == 0 else half_kernel
            y2_read = ((y2_in - y1_in) if y2_in == height else (y2_in - y1_in) - half_kernel)
            x1_read = 0 if x1_in == 0 else half_kernel
            x2_read = ((x2_in - x1_in) if x2_in == width else (x2_in - x1_in) - half_kernel)

            # Write the valid data
            output[:, y1_out:y2_out, x1_out:x2_out] = p_output[
                :, y1_read:y2_read, x1_read:x2_read
            ]

    if nodata is not None:
        output[~valid_mask] = 0

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

            # Skip chunks where all bands have zero stdev (e.g., all nodata)
            if np.all(stdevs == 0):
                continue

            # Avoid division by zero for individual bands with zero stdev
            stdevs = np.where(stdevs == 0, 1, stdevs)

            deviations[:, y : y + step_size, x : x + step_size] = (
                chunk - means
            ) / stdevs

            # set nodata source pixels to deviation=0 since `data` array includes nodata as a value
            if nodata is not None:
                deviations[:, y : y + step_size, x : x + step_size] = np.where(
                    chunk != nodata,
                    deviations[:, y : y + step_size, x : x + step_size],
                    0
                )

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
