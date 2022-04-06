# P3Depth: Monocular Depth Estimation with a Piecewise Planarity Prior

> **P3Depth: Monocular Depth Estimation with a Piecewise Planarity Prior**
>
> [Vaishakh Patil](https://www.trace.ethz.ch/team/members/vaishakh.html), [Dr. Christos Sakaridis](https://people.ee.ethz.ch/~csakarid/), [Dr. Alex Liniger](https://www.trace.ethz.ch/team/members/alex.html) and [Prof. Luc Van Gool](https://www.trace.ethz.ch/team/members/luc.html)
>
> CVPR 2022 (pdf) coming soon...
> 
> [Arxiv](https://arxiv.org/abs/2204.02091) ([pdf](https://arxiv.org/pdf/2204.02091v1.pdf))


## Abstract
<p align="center">
  <img src="assets/teaser.png" alt="example input output" width="1000" />
</p>

Monocular depth estimation is vital for scene understanding and downstream tasks. We focus on the supervised setup, in which ground-truth depth is available only at training time. Based on knowledge about the high regularity of real 3D scenes, we propose a method that learns to selectively leverage information from coplanar pixels to improve the predicted depth. In particular, we introduce a piecewise planarity prior which states that for each pixel, there is a seed pixel which shares the same planar 3D surface with the former. Motivated by this prior, we design a network with two heads. The first head outputs pixel-level plane coefficients, while the second one outputs a dense offset vector field that identifies the positions of seed pixels. The plane coefficients of seed pixels are then used to predict depth at each position. The resulting prediction is adaptively fused with the initial prediction from the first head via a learned confidence to account for potential deviations from precise local planarity. The entire architecture is trained end-to-end thanks to the differentiability of the proposed modules and it learns to predict regular depth maps, with sharp edges at occlusion boundaries. An extensive evaluation of our method shows that we set the new state of the art in supervised monocular depth estimation, surpassing prior methods on NYU Depth-v2 and on the Garg split of KITTI. Our method delivers depth maps that yield plausible 3D reconstructions of the input scenes.

## 📖 Citations

If you find our work useful in your research please consider citing our publication:

```bibtex
@inproceedings{P3Depth,
  author    = {Patil, Vaishakh and Sakaridis, Christos and Liniger, Alex and Van Gool, Luc},
  title     = {P3Depth: Monocular Depth Estimation with a Piecewise Planarity Prior},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022},
}
```
