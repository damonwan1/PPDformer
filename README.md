
# PPDformer: Channel-Specific Periodic Patch Division for Time Series Forecasting

This repository contains the implementation of **PPDformer**, a novel deep learning model designed for multivariate time series forecasting (MTS). 

## Overview

The PPDformer addresses critical challenges in MTS forecasting:
- **Channel-Independent Denoising**: It applies adaptive noise cancellation and Fourier Transform techniques to individually denoise each channel. This improves accuracy by preserving important details and removing noise more effectively.
- **Periodic Patch Division**: It segments time series data based on periodic patterns detected per channel. This segmentation allows for better identification of temporal dependencies and local information.

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/PPDformer.git
cd PPDformer
pip install -r requirements.txt
```

## Model Architecture

PPDformer is composed of the following major modules:
- **CI Denoising**: This module denoises each channel using FFT and adaptive noise cancellation.
- **Periodic Patch Division**: It aligns the time series data by clustering periods and then converting them into two-dimensional patches.
- **Dual Attention Mechanism**: Combines patch attention for fine-grained detail capture with full attention for global dependency modeling.

For detailed architecture explanations, please refer to our paper.


## License

This project is licensed under the APache 2.0 License - see the LICENSE file for details.

## Acknowledgements

We would like to thank all contributors and collaborators for their input and support.
