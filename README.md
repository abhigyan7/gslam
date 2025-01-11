# GS SLAM
Monocular SLAM using Gaussian Splatting.

## Setting up
- Create an env from the spec file

``` sh
conda create -f environment.yml
```

- Install [https://github.com/nerfstudio-project/gsplat](gsplat)
- Install [https://github.com/nerfstudio-project/nerfacc](nerfacc) (don't if you won't be running gsplat tests)
- Download a sequence (or more) from the TUM-rgbd dataset.
- Run 'python main.py'
