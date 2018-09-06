// This file is part of otmap, an optimal transport solver.
//
// Copyright (C) 2018 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "generic_tasks.h"
#include "image_utils.h"
#include "analytical_functions.h"
#include "utils/rasterizer.h"
#include "utils/stochastic_rasterizer.h"
#include "utils/eigen_addons.h"

using namespace Eigen;
using namespace surface_mesh;
using namespace otmap;

bool load_input_density(const std::string& filename, MatrixXd& density)
{
  if(filename[0]==':')
  {
    // procedural density function
    int func_id;
    int resolution;
    if(std::sscanf(filename.c_str(), ":%d:%d:", &func_id, &resolution)!=2)
    {
      std::cerr << "Error parsing procedural input \"" << filename << "\", expected format is \":id:resolution:\"";
      return false;
    }
    density.resize(resolution,resolution);
    eval_func_to_grid(density, func_id);
  }
  else
  {
    load_image(filename.c_str(), density);
    if(density.size()==0)
      return false;
    if(density.rows()!=density.cols())
    {
      std::cout << "Error: input image \"" << filename << "\" is not square.";
      return false;
    }
  }
  return true;
}

void generate_transport_maps(const std::vector<std::string>& inputs, std::vector<TransportMap>& tmaps, const CLI_OTSolverOptions& opts,
                            std::function<void(Eigen::MatrixXd&)> filter)
{
  GridBasedTransportSolver otsolver;
  otsolver.set_verbose_level(opts.verbose_level);

  if(opts.verbose_level>=1)
    std::cout << "Generate all transport maps...\n";
  for(int k=0; k<inputs.size(); ++k)
  {
    MatrixXd density;
    if(!load_input_density(inputs[k], density))
    {
      std::cout << "Failed to load input #" << k << " \"" << inputs[k] << "\" -> abort.";
      exit(EXIT_FAILURE);
    }
    otsolver.init(density.rows());
    filter(density);
    tmaps.push_back( otsolver.solve(vec(density), opts.solver_opt) );
  }
}

void synthetize_and_export_image(const Surface_mesh& map, int img_res, const VectorXd& target, const std::string base_filename, const VectorXd& input_density, double gamma)
{
 int samples_per_face = 300;

  MatrixXd img(img_res,img_res);
  if(input_density.size()>0) {
    VectorXi spf = (samples_per_face*input_density.size()/input_density.sum()*input_density).cast<int>();
    if(map.faces_size() == 2*spf.size()) {
      // grid represented as triangles, duplicate entries:
      spf.conservativeResize(map.faces_size());
      for(int i=map.faces_size()-1; i>=0; i--)
        spf(i) = spf(i/2);
    }
    sample_transportmap_to_image(map, spf, img);
  } else {
    sample_transportmap_to_image(map, img, samples_per_face);
  }

  double Esource  = /*target.size() == map.faces_size() ? target.sum()*0.5 : */target.sum();
  double Eimg     = double(samples_per_face * map.faces_size());

  img = img * Esource/Eimg ;

  save_image(std::string(base_filename).append("_sampling.bmp").c_str(), img.array().pow(gamma));

  rasterize_image(map, img);
  img = img * target.sum() / target.size();
  save_image(std::string(base_filename).append("_raster.bmp").c_str(), img.array().pow(gamma));
}
