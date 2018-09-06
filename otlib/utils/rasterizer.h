// This file is part of otmap, an optimal transport solver.
//
// Copyright (C) 2017-2018 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <Eigen/Core>
#include <surface_mesh/Surface_mesh.h>

namespace otmap {

enum RasterImageOption {
  RIO_PerVertexDensity,
  RIO_PerFaceDensity
};

void rasterize_image(const surface_mesh::Surface_mesh& mesh, const Eigen::VectorXd &density_per_Face, Eigen::MatrixXd& img, RasterImageOption opt = RIO_PerFaceDensity);
void rasterize_image(const surface_mesh::Surface_mesh& mesh, Eigen::MatrixXd& img, RasterImageOption opt = RIO_PerFaceDensity);

} // namespace otmap
