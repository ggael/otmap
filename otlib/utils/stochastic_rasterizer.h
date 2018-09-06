// This file is part of otmap, an optimal transport solver.
//
// Copyright (C) 2016-2018 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <Eigen/Core>
#include <surface_mesh/Surface_mesh.h>
#include "transport_map.h"

namespace otmap {

void sample_transportmap_to_image(const surface_mesh::Surface_mesh& mesh, const Eigen::VectorXi &sample_per_face, Eigen::MatrixXd& img);

void sample_transportmap_to_image(const surface_mesh::Surface_mesh& mesh, Eigen::MatrixXd& img, int sample_per_face = 100);

} // namespace otmap
