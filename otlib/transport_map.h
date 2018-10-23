// This file is part of otmap, an optimal transport solver.
//
// Copyright (C) 2018 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include <Eigen/Core>
#include "surface_mesh/Surface_mesh.h"
#include <memory>

namespace otmap
{

class BVH2D;

class TransportMap
{
public:

  TransportMap( std::shared_ptr<surface_mesh::Surface_mesh> origin_mesh,
                std::shared_ptr<surface_mesh::Surface_mesh> fwd_mesh,
                std::shared_ptr<Eigen::VectorXd> density);
  TransportMap(const TransportMap& other) = default;

  ~TransportMap();

  /** this function must be called at least once before calling inv/inv_fast */
  void init_inverse() const;

  Eigen::Vector2d fwd(const Eigen::Vector2d& p) const;
  Eigen::Vector2d inv(const Eigen::Vector2d& p) const { return inv_impl(p,false); }
  Eigen::Vector2d inv_fast(const Eigen::Vector2d& p) const { return inv_impl(p,true); }

  std::shared_ptr<surface_mesh::Surface_mesh> fwd_mesh_ptr() { return m_fwd_mesh; }
  std::shared_ptr<Eigen::VectorXd> density_ptr() { return m_density; }

  const surface_mesh::Surface_mesh& origin_mesh() const { return *m_origin_mesh; }
  const surface_mesh::Surface_mesh& fwd_mesh() const { return *m_fwd_mesh; }
  const Eigen::VectorXd& density() const { return *m_density; }


protected:

  Eigen::Vector2d inv_impl(const Eigen::Vector2d& p, bool fast_mode) const;

  std::shared_ptr<surface_mesh::Surface_mesh> m_origin_mesh;
  std::shared_ptr<surface_mesh::Surface_mesh> m_fwd_mesh;
  std::shared_ptr<Eigen::VectorXd> m_density;
  mutable BVH2D* m_bvh;
};

/** Inverts uniform mesh relative to a transport map */
void apply_inverse_map( const otmap::TransportMap& tmap,
                        std::vector<Eigen::Vector2d> &points, /* in-out */
                        int verbose_level = 2);

double transport_cost(const surface_mesh::Surface_mesh &src_mesh, const surface_mesh::Surface_mesh &dst_mesh, const Eigen::VectorXd &density_per_face, Eigen::VectorXd *cost_per_face = 0);

} // namespace otmap
