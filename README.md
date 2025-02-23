# GS SLAM
Monocular SLAM using Gaussian Splatting.

## Setting up
- Create an env from the spec file

``` sh
conda env create -f environment.yml
```

- Install [https://github.com/abhigyan7/gsplat/tree/gslam](abhigyan7/gsplat/tree/gslam)
- Install [https://github.com/rahul-goel/fused-ssim](rahul-goel/fused-ssim)

``` python
pip install git+https://github.com/rahul-goel/fused-ssim@30fb258c8a38fe61e640c382f891f14b2e8b0b5a
```

- Install [https://github.com/nerfstudio-project/nerfacc](nerfacc) (don't if you won't be running gsplat tests)
- Run `pre-commit install` to set up the git hooks for formatting and linting
- Download a sequence (or more) from the TUM-rgbd dataset
- Run 'python main.py'
