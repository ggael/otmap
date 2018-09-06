// This file is part of otmap, an optimal transport solver.
//
// Copyright (C) 2016-2018 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "stochastic_rasterizer.h"
#include <vector>
#include <Eigen/Core>
#include <surface_mesh/Surface_mesh.h>

using namespace surface_mesh;
using namespace Eigen;

namespace otmap {

void sample_transportmap_to_image(const Surface_mesh &mesh, const VectorXi &sample_per_face, MatrixXd& img)
{
  int rows = img.rows();
  int cols = img.cols();
  double dx = 1./double(cols);
  double dy = 1./double(rows);

  img.setZero();

  std::vector<Vector3d> samples;

  int nf = mesh.faces_size();
  for(int i=0; i<nf; ++i){
    int ns = sample_per_face(i);
    if(ns>0) {
      Surface_mesh::Face f(i);

      Surface_mesh::Vertex indices[4];
      int nv = 0;
      for(auto v:mesh.vertices(f))
        indices[nv++] = v;

      Vector2d v1 = mesh.position( indices[0] );
      Vector2d v2 = mesh.position( indices[1] );
      Vector2d v3 = mesh.position( indices[2] );
      Vector2d v4;
      if(nv==4)
        v4 = mesh.position( indices[3] );

      assert(nv==3 || nv==4);

      samples.resize(ns);
      if(nv==3)
        for(int j=0; j<ns; ++j) {
          double r1 = std::sqrt(internal::random<double>(0.,1.));
          double r2 = internal::random<double>(0.,1.);
          samples[j] << 1. - r1, r1*(1.-r2), r1*r2;
        }
      else
      {
        for(int j=0; j<ns; ++j)
          samples[j] = Vector3d::Random().cwiseAbs();
      }
      for(int j = 0; j<ns; ++j)
      {
        //compute position
        Eigen::Vector2d p;
        if(nv==3)
          p = samples[j](0)*v1 + samples[j](1)*v2 + samples[j](2)*v3;
        else
        {
          double u = samples[j](0);
          double v = samples[j](1);
          p = (1.-u)*(1-v)*v1 + (u)*(1-v)*v2 + (u)*(v)*v3 + (1.-u)*(v)*v4;
        }
        if(p.x()<0 || p.y()<0 || p.x()>1 || p.y()>1 || !p.array().isFinite().all()) {
          if(!p.array().isFinite().all())
            std::cout << samples[j].transpose().head<2>() << "   ;   " << p.transpose() << " " << v1.transpose() << " " << v2.transpose() << " " << v3.transpose() << " " << v4.transpose() << " " << "\n";
          continue;
        }
        //compute the pixel
        int pi = int(p.y()/dy);
        int pj = int(p.x()/dx);

        if(pi < rows && pj<cols)
          img(pi,pj) += 1.;
      }
    }
  }
}

void sample_transportmap_to_image(const surface_mesh::Surface_mesh& mesh, Eigen::MatrixXd& img, int sample_per_face)
{
  sample_transportmap_to_image(mesh, Eigen::VectorXi::Constant(mesh.faces_size(), sample_per_face), img);
}

} // namespace otmap
