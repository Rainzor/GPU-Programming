CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

# Features
- **Wavefront Path Tracing**
  - **Sperating megakernels into several kernels**: `RayGen`, `Intersection`, `Shade` ...
  - **Less Warp Divergence**: Stream compacting the ray path segments 
  - **Improve Memory Access Coalescing**: SoA data layout and sorting path segments by material type before shading
- **Texture Mapping**: More surface color based on CUDA `Texture Object`
- **Triangle Primitive Mesh**: More shapes based on `Obj` and `glTF` loader 
- **Accelerate Data Structure** `BVH`: 
  - Binary Radix Tree Construction
  - Parallel Tree Traversal
  - Two Level Acceleration Structures: at the world level (TLAS) and at the model level (BLAS)
- **Efficient Monte Carlo Integrator**: 
  - **Next Event Estimation**: Shadow rays are explicitly aimed at light sources to reduce variance.
  - **Multiple Importance Sampling**: Combining the PDFs of different sampling techniques to reduce variance.
- **Physically Based Rendering**:
  - Diffuse (Lambertian)
  - Fresnel Reflection
  - Microfacet BRDF
- **Participating Media**:
  - Homogeneous Medium

# Reference
- [CIS 5650 GPU Programming and Architecture](https://cis5650-fall-2024.github.io/)
- [Physically Based Rendering V4](https://www.pbr-book.org/4ed/contents)
- [GPU-Raytracer (Open Source Code)](https://github.com/jan-van-bergen/GPU-Raytracer)

