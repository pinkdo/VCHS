# VCHS

MATLAB filtering core for **Vectorial Coupling and Hessian Spectral Sparsity for Anisotropic Edge-Preserving Smoothing**.

The method combines a joint color-gradient penalty with a channel-wise Hessian spectral penalty. An ADMM iteration uses FFT-based image updates and magnitude-dependent p-shrinkage for the auxiliary variables. The filtered image can serve as a smoothed base layer for detail enhancement or be used for JPEG artifact reduction with application-specific parameters.

## Files

| File | Purpose |
| --- | --- |
| `main.m` | Self-contained CPU filtering implementation. |

## Requirements

- MATLAB with Image Processing Toolbox (`psf2otf`).

## Quick start

Clone the repository and change MATLAB's current folder to the cloned directory, or add that directory to the MATLAB path:

```sh
git clone https://github.com/pinkdo/VCHS.git
```

Supply an input image and run:

```matlab
y = im2double(imread('input.png'));

% Example configuration.
params = struct('lambda1', 0.01243, 'lambda2', 0.0004662, ...
                'rho1', 2.43857, 'rho2', 0.21213, ...
                'p1', 0.0138, 'p2', 0.1085, ...
                'max_iter', 100);

x = main(y, params);

% Clip for display or image export; x retains the raw solver output.
x_display = min(max(x, 0), 1);
imshow(x_display);
imwrite(x_display, 'filtered.png');
```

These parameters demonstrate the interface. Select parameters for the intended input scale and application.

## Input and output

```matlab
x = main(y, params)
```

- `y`: a real double image of size `H`-by-`W` or `H`-by-`W`-by-`C`, with `H,W >= 3`. RGB images use `C = 3`; grayscale images are also supported. The example uses intensities in `[0,1]`.
- `params`: a structure containing all seven required fields below.
- `x`: the filtered image, with the same size as `y`.

The caller supplies intensity conversion and any application-specific preprocessing. The core processes the supplied image directly and returns floating-point values without clipping.

## Parameters

| Field | Meaning | Intended range |
| --- | --- | --- |
| `lambda1` | First-order regularization weight | Nonnegative |
| `lambda2` | Hessian spectral regularization weight | Nonnegative |
| `rho1` | First-order quadratic splitting coefficient | Positive |
| `rho2` | Second-order quadratic splitting coefficient | Positive |
| `p1` | First-order shrinkage shape parameter | `0 < p1 <= 1` |
| `p2` | Hessian shrinkage shape parameter | `0 < p2 <= 1` |
| `max_iter` | Number of ADMM iterations | Positive integer |

The shrinkage scales are `lambda1/rho1` and `lambda2/rho2`. The quadratic coefficients also enter the FFT image update.

## Implementation details

- Spatial derivatives use periodic boundaries through FFTs.
- The first-order update couples all image channels through a joint gradient magnitude at each pixel.
- The second-order update operates on a symmetric 2-by-2 Hessian independently in each channel, shrinking its signed eigenvalues and reconstructing the matrix. The mixed Hessian component has weight 2 in the associated quadratic terms.
- Initialization uses the input image and zero auxiliary and scaled dual fields.
- The solver performs exactly `max_iter` iterations.
- The p-shrinkage updates follow the implicit-penalty construction described in the paper.

This repository contains the filtering implementation and usage instructions. Full experimental reproduction additionally requires the corresponding datasets, preprocessing, parameter settings, and evaluation scripts.
