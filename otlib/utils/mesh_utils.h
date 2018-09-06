// This file is part of otmap, an optimal transport solver.
//
// Copyright (C) 2017-2018 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <surface_mesh/Surface_mesh.h>
#include <Eigen/Core>

#include "eigen_addons.h"

namespace otmap {
  
inline Eigen::Vector2d bilinear_coordinates_in_triangle(const Eigen::Vector2d& q, const Eigen::Vector2d& p0, const Eigen::Vector2d& p1, const Eigen::Vector2d& p2)
{
  Eigen::Vector2d q2 = q  - p2;
  Eigen::Vector2d eu = p1 - p2;
  Eigen::Vector2d ev = p0 - p2;
  double area2 = cross2(ev,eu);
  return Eigen::Vector2d(cross2(q2,eu)/area2, cross2(ev,q2)/area2);
}

bool bilinear_coordinates_in_quad(const Eigen::Vector2d& q, const Eigen::Vector2d *p, double &u, double &v);
bool bilinear_coordinates_in_quad(const Eigen::Vector2d& q, const Eigen::Vector2d *p, Eigen::Ref<Eigen::Vector4d> w);

bool inside_quad(const Eigen::Vector2d& q, const Eigen::Vector2d *p);

// generate a regular quad mesh
void generate_quad_mesh(int m, int n, surface_mesh::Surface_mesh& mesh, bool inclusive = false);

// removes faces having a zero density
void prune_empty_faces(surface_mesh::Surface_mesh &mesh, Eigen::VectorXd density);

} // namespace otmap

