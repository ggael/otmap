# OTMap: a c++ solver for computing (Optimal) Transport Maps on 2D grids

This is a lightweight implementation of "Instant Transport Maps on 2D Grids" [1].

It currently supports L2-optimal maps from an arbitrary density defined on a uniform 2D grid (aka an image) to a square with uniform density.
Inverse maps and maps between pairs of arbitrary images are then recovered through numerical inversion and composition resulting in density preserving but approximately optimal maps.

This code comes with 3 demos:
- otmap: computes the forward and backward maps between one image and a uniform square or between a pair of images. The maps are exported as .off quad meshes.
- stippling: adapt a uniformly distributed point cloud to a given image.
- barycenters: computes linear (resp. bilinear) approximate Wasserstein barycenters between a pair (resp. four) images.

## Examples

#### bilinear barycenters

```
$ ./barycenters -in ../data/shape?instant.png -blur 5 -steps 4 -inv -img_eps 1e-3 -th 1e-7 -out shape
```
![bilinear barycenters](https://github.com/ggael/otmap/master/doc/bilinear_barycenters.png "Bilinear barycenters")

#### sampling

```
$ ./stippling -in ../data/julia.png
```
![stippling](https://github.com/ggael/otmap/master/data/sampling_julia.png "stippling")

## Installation

This code uses [Eigen](https://eigen.tuxfamily.org), Surface_mesh, and CImg that are already included in the repo/archive.
It is however highly recommended to install [SuiteSparse/Cholmod](http://faculty.cse.tamu.edu/davis/suitesparse.html) for higher performance, and libpng/libjpg for image IO.

All you need is to clone the repo, configure a build directory with cmake, and then build.
For instance:

````
$ git clone ...
$ cd otmap
$ mkdir build
$ cd build
$ cmake ..
$ make -j 8
````

## License

The core of the transport solver is provided under the [GNU Public License v3](https://www.gnu.org/licenses/gpl-3.0.html).

Utilities and applications are released under the [Mozilla Public License 2](https://www.mozilla.org/en-US/MPL/2.0/).

## References

[1] Georges Nader and Gael Guennebaud. _Instant Transport Maps on 2D Grids_. ACM Transactions on Graphics (Proceedings of Siggraph Asia 2018).