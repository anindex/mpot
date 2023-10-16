# Accelerating Motion Planning via Optimal Transport

This repository implements Motion Planning via Optimal Transport `mpot` in PyTorch. 
The philosophy of `mpot` follows Monte Carlo methods' argument, i.e., more samples could discover more better modes with high enough initialization variances.
In other words, within multi-modal motion planning scope, `mpot` enables better **brute-force** planning with GPU vectorization. This enhances robustness against bad local minima, which is common in optimization-based motion planning.

<p float="middle">
  <img src="demos/occupancy.gif" width="32%" />
  <img src="demos/sdf_grid.gif" width="32%" /> 
  <img src="demos/panda.gif" width="32%" />
</p>

## Installation

Simply activate your conda/Python environment and run

```azure
pip install -e .
```

`mpot` algorithm is specifically designed to work with GPU. Please check if you have installed PyTorch with the CUDA option. 

## Examples

Please find in `examples/` folder the demo of vectorized planning in planar environments with occupancy map:

```azure
python examples/mpot_occupancy.py
```

and with signed-distance-field (SDF):

```azure
python examples/mpot_sdf.py
```

We also added a demo with vectorized Panda planning with dense obstacle environments (SDF):

```azure
python examples/mpot_panda.py
```

Every run is associated with **a different seed**. The resulting optimization visualizations are stored at your current directory.
Please refer to the example scripts for playing around with options and different goal points. Note that for all cases, we normalize the joint space to the joint limits and velocity limits, then perform Sinkhorn Step on the normalized state-space. Changing any hyperparameters may require tuning again.

**Tuning Tips**: The most sensitive parameters are:

- `polytope`: for small state-dimension that is less than 10, `cube` is a good choice. For much higer state-dimension, the sensible choices are `orthoplex` or `simplex`.
- `step_radius`: the step size.
- `probe_radius`: the probing radius, which projects towards polytope vertices to compute cost-to-go. Note, `probe_radius` >= `step_radius`.
- `num_probe`: number of probing points along the probe radius. This is critical for optimizing performance, usually 3-5 is enough.
- `epsilon`: decay rate of the step/probe size, usually 0.01-0.05.
- `ent_epsilon`: Sinkhorn entropy regularization, usually 1e-2 to 5e-2 for balancing between optimal coupling's sharpness and speed.
- Various cost term weightings. This depends on your applications.

## Troubleshooting

If you encounter memory problems, try:

```azure
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
```

to reduce memory fragmentation.

## Citation

If you found this repository useful, please consider citing these references:

```azure
@inproceedings{le2023accelerating,
  title={Accelerating Motion Planning via Optimal Transport},
  author={Le, An T. and Chalvatzaki, Georgia and Biess, Armin and Peters, Jan},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
