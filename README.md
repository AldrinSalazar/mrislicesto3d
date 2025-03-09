# MRISlicesTo3D

<p align="center">
  <img src="https://img.shields.io/badge/MRI-3D%20Reconstruction-brightgreen" alt="MRI 3D Reconstruction">
  <img src="https://img.shields.io/badge/Status-Stable-green" alt="Status: Stable">
  <img src="https://img.shields.io/badge/License-MIT-blue" alt="License: MIT">
</p>

## Overview

MRISlicesTo3D is a high-performance Go implementation for reconstructing 3D volumetric images from 2D MRI slices. It implements advanced algorithms for medical image processing with a focus on accuracy and computational efficiency through parallel processing.

This project is based on the methodology described in the research paper "Fast 3D Volumetric Image Reconstruction from 2D MRI Slices by Parallel Processing" by Somoballi Ghoshal et al.

## Key Features

- **High-Performance Reconstruction**: Achieves ~70% speedup through parallel processing
- **Edge-Preserved Interpolation**: Utilizes kriging interpolation with 3D neighborhood consideration
- **Advanced Edge Detection**: Implements shearlet transform with orientation-based analysis
- **Accurate Reconstruction**: Achieves ~98.9% accuracy for spine datasets and ~99.0% for brain datasets
- **Optimized for Multi-Core Systems**: Automatic workload distribution across available CPU cores
- **3D Model Generation**: Creates STL files using marching cubes algorithm
- **Comprehensive Validation**: Implements metrics from the research paper (MI, RMSE, SSIM)

## Technical Implementation

The implementation focuses on these key algorithms from the paper:

1. **Edge-preserved Kriging Interpolation**
   - Maintains spatial correlation between slices
   - Preserves critical structural details in medical images

2. **Shearlet Transform**
   - Multi-scale, multi-directional edge detection
   - Orientation-based feature preservation

3. **Parallel Data Processing**
   - Input data subdivision into 8 equal parts
   - SIMD-like execution across multiple cores

4. **3D Volume Reconstruction**
   - Marching cubes algorithm for surface extraction
   - Proper scaling based on inter-slice spacing

## Installation

### Prerequisites
- Go 1.16 or later

### Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mrislicesto3d.git
cd mrislicesto3d
```

2. Install dependencies:
```bash
go mod download
```

3. Build the project:
```bash
go build ./cmd/mrislicesto3d
```

## Usage

### Basic Command

```bash
./mrislicesto3d -input /path/to/slices -output output.stl -gap 3.0 -cores 8 -isolevel 0.25 -edge-threshold 0.5
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-input`  | Directory containing the 2D MRI slices | *Required* |
| `-output` | Output STL filename for the 3D model | `output.stl` |
| `-gap`    | Inter-slice gap in millimeters | `3.0` |
| `-cores`  | Number of CPU cores to use | All available |
| `-isolevel` | IsoLevel percent for volume generation (0.0-1.0) | `0.25` |
| `-edge-threshold` | Edge detection threshold (0.0-1.0) | `0.5` |

## Performance Metrics

Based on the methodology in the research paper:

| Metric | Performance |
|--------|-------------|
| Speed Improvement | ~70% faster with 8 cores vs. single-core |
| Accuracy (Spine) | ~98.9% |
| Accuracy (Brain) | ~99.0% |
| Edge Preservation | High fidelity to original structures |

## Validation Methods

The implementation includes these validation metrics:

- Mutual Information (MI)
- Entropy Difference
- Root Mean Square Error (RMSE)
- Structural Similarity Index (SSIM)
- Edge Preservation Ratio

## Current Limitations

- Supports JPEG images only (not DICOM)
- Outputs STL format only
- Simplified implementation of shearlet transform compared to the paper
- No integrated visualization interface

## Citation

If you use this implementation in your research, please cite the original paper:

```
@article{ghoshal2023fast,
  title={Fast 3D Volumetric Image Reconstruction from 2D MRI Slices by Parallel Processing},
  author={Ghoshal, Somoballi and Goswami, Shremoyee and Chakrabarti, Amlan and Sur-Kolay, Susmita},
  journal={arXiv preprint arXiv:2303.09523},
  year={2023}
}
```

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request 