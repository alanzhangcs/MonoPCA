<div align="center">

# Rethinking Geometric Depth in Monocular 3D Object Detection: A Projection-Consistent Reformulation

[Zhihao Zhang](https://alanzhangcs.github.io/)<sup>1</sup>, [Abhinav Kumar](https://sites.google.com/view/abhinavkumar)<sup>1</sup>, [Huaizhi Qu](https://qhz991029.github.io/)<sup>2</sup>, [Tianlong Chen](https://tianlong-chen.github.io/)<sup>2</sup>, [Xiaoming Liu](https://cs.unc.edu/person/xiaoming-liu/)<sup>1,2</sup>

<sup>1</sup>Michigan State University &nbsp; <sup>2</sup>University of North Carolina at Chapel Hill

**NeurIPS 2026**

[![Project Page](https://img.shields.io/badge/Project-Page-4B9CD3)](https://alanzhangcs.github.io/MonoPCA-Page/)
![Venue](https://img.shields.io/badge/NeurIPS-2026-13294B)

</div>

<p align="center">
  <img src="assets/framework.png" width="100%" alt="Overview of MonoPCA">
</p>

**MonoPCA** shows that the widely used geometric depth $z = f \cdot H / h_{2D}$ recovers the depth of an object's visible front surface rather than its center. **PCDepth** corrects this bias in closed form with an orientation- and size-aware term, and **GeoAlign** distills a Vision Foundation Model teacher into a lightweight student to sharpen the size and yaw predictions PCDepth relies on, at no inference cost.

## Code

🚧 **The code will be released soon.** Please star or watch this repository for updates.

## Citation

```bibtex
@inproceedings{zhang2026rethinking,
  title     = {Rethinking Geometric Depth in Monocular 3D Object Detection:
               A Projection-Consistent Reformulation},
  author    = {Zhang, Zhihao and Kumar, Abhinav and Qu, Huaizhi and
               Chen, Tianlong and Liu, Xiaoming},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
