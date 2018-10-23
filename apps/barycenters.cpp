// This file is part of otmap, an optimal transport solver.
//
// Copyright (C) 2017-2018 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2017 Georges Nader
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <iomanip>
#include <vector>

#include <surface_mesh/Surface_mesh.h>
#include "otsolver_2dgrid.h"
#include "common/otsolver_options.h"
#include "utils/rasterizer.h"
#include "common/image_utils.h"
#include "common/generic_tasks.h"

using namespace Eigen;
using namespace surface_mesh;
using namespace otmap;

void output_usage()
{
  std::cout << "usage : otsolver <option> <value>" << std::endl;

  std::cout << "input options:" << std::endl;
  std::cout << " * -in i0 i1 [i2 i3] where i* is either a <filename> or a procedural func \":id:res:\"" << std::endl;
  std::cout << " *                   see analytical_functions.h for a list of possible function ids." << std::endl;

  std::cout << "morphing options :" << std::endl;
  std::cout << " * -steps <nsteps>     (default: 4)" << std::endl;
  std::cout << " * -blur <diameter>    (default: 0)" << std::endl;
  std::cout << " * -img_eps <value>    (default: 1e-5)" << std::endl;
  std::cout << " * -inv                use 1-img" << std::endl;
  std::cout << " * -export_maps        write maps as .off files" << std::endl;

  CLI_OTSolverOptions::print_help();

  std::cout << "output options :" << std::endl;
  std::cout << " * -out <prefix>" << std::endl;
}

struct CLIopts : CLI_OTSolverOptions
{
  std::vector<std::string> inputs;
  std::string out_prefix;

  int nsteps;
  double blur_scale;
  double img_eps;
  bool use_inv;
  bool export_maps;

  void set_default()
  {
    inputs.clear();

    nsteps = 4;

    out_prefix = "";
    blur_scale = 0;
    img_eps = 1e-5;
    use_inv = false;
    export_maps = false;

    CLI_OTSolverOptions::set_default();
  }

  bool load(const InputParser& args)
  {
    set_default();

    CLI_OTSolverOptions::load(args);

    std::vector<std::string> value;

    if(args.getCmdOption("-in", value))
    {
      inputs = value;
    }
    else
    {
      std::cerr << "missing -in filename or -in :id:res: to specify input" << std::endl;
      return false;
    }

    if(args.getCmdOption("-steps", value))
      nsteps = std::stoi(value[0]);

    if(args.getCmdOption("-blur", value))
      blur_scale = std::stof(value[0]);

    if(args.getCmdOption("-img_eps", value))
      img_eps = std::stof(value[0]);

    if(args.getCmdOption("-out", value))
      out_prefix = value[0];
    
    if(args.cmdOptionExists("-inv"))
      use_inv = true;

    if(args.cmdOptionExists("-export_maps"))
      export_maps = true;

    return true;
  }
};



void interpolate(const std::vector<Surface_mesh> &inv_maps, double alpha, double beta, Surface_mesh& result);
void interpolate(const std::vector<Surface_mesh> &inv_maps, double alpha, Surface_mesh& result);
void synthetize_and_save_image(const Surface_mesh& map, const std::string& filename, int res, double expected_mean, bool inv);

std::string
make_padded_string(int n, int nzero = 3)
{
  std::stringstream ss;
  ss << "_";
  ss << std::setfill('0') << std::setw(nzero) << n;
  return ss.str();
}

template<typename T,typename S>
T bilerp(S u, S v, const T& a0, const T& a1, const T& a2, const T& a3)
{
  return (1.-u)*(1.-v)*a0
       + (   u)*(1.-v)*a1
       + (   u)*(   v)*a2
       + (1.-u)*(   v)*a3;
}

template<typename T,typename S>
T lerp(S u, const T& a0, const T& a1)
{
  return (1.-u)*a0 + u*a1;
}

int main(int argc, char** argv)
{
  InputParser input(argc, argv);

  if(input.cmdOptionExists("-help") || input.cmdOptionExists("-h")){
    output_usage();
    return 0;
  }

  CLIopts opts;
  if(!opts.load(input)){
    std::cerr << "invalid input" << std::endl;
    output_usage();
    return 1;
  }

  if(opts.inputs.size()!=2 && opts.inputs.size()!=4)
  {
    std::cerr << "Got " << opts.inputs.size() << " input densities, but you must provide either 2 or 4.\n";
    output_usage();
    return 1;
  }

  std::vector<TransportMap> tmaps;
  std::vector<MatrixXd> input_densities;
  generate_transport_maps(opts.inputs, tmaps, opts,
    [&opts,&input_densities](MatrixXd& img) {
      if(opts.use_inv)
        img = 1.-img.array();
      
      img.array() += opts.img_eps;
      
      if(opts.blur_scale>0) {
        MatrixXd tmp(img.rows(), img.cols());
        gaussian_blur(img, tmp, opts.blur_scale);
        img.swap(tmp);
      }

      input_densities.push_back(img);
    });

  std::vector<Surface_mesh> inv_maps(tmaps.size());
  int img_res = input_densities[0].rows();
  std::cout << "Generate inverse maps...\n";
  std::vector<double> density_means(tmaps.size());
  for(int k=0; k<tmaps.size(); ++k)
  {
    inv_maps[k] = tmaps[k].origin_mesh();
    apply_inverse_map(tmaps[k], inv_maps[k].points(), opts.verbose_level);
    density_means[k] = input_densities[k].mean();

    if(opts.export_maps) {
      tmaps[k].fwd_mesh().write(std::string(opts.out_prefix).append("_fwd_").append(make_padded_string(k)).append(".off"));
      inv_maps[k].write(std::string(opts.out_prefix).append("_inv_").append(make_padded_string(k)).append(".off"));
    }
  }

  std::cout << "Interpolate...\n";
  for(int i=0; i<=opts.nsteps; ++i)
  {
    double alpha = double(i)/double(opts.nsteps);

    if(tmaps.size()==4)
    {
      // bilinear barycenters
      for(int j=0; j<=opts.nsteps; ++j)
      {
        Surface_mesh res;

        std::string filename = opts.out_prefix;
        filename.append(make_padded_string(i)).append("_").append(make_padded_string(j)).append(".png");

        
        double beta  = double(j)/double(opts.nsteps);
        interpolate(inv_maps, alpha, beta, res);
        double expected_mean = bilerp(alpha, beta, density_means[0], density_means[1], density_means[2], density_means[3]);
        synthetize_and_save_image(res, filename, img_res, expected_mean, opts.use_inv);

        if(opts.export_maps)
          res.write(std::string(opts.out_prefix).append("_alpha_map_").append(make_padded_string(i)).append("_").append(make_padded_string(j)).append(".off"));
        
        std::cout << i*(opts.nsteps+1)+j+1 << "/" << (opts.nsteps+1)*(opts.nsteps+1) << "\r" << std::flush;
      }
    }
    else // if(tmaps.size()==2)
    {
      assert(tmaps.size()==2);
      Surface_mesh res;

      std::string filename = opts.out_prefix;
      filename.append(make_padded_string(i)).append(".png");

      interpolate(inv_maps, alpha, res);
      double expected_mean = lerp(alpha, density_means[0], density_means[1]);
      synthetize_and_save_image(res, filename, img_res, expected_mean, opts.use_inv);

      if(opts.export_maps)
        res.write(std::string(opts.out_prefix).append("_alpha_map_").append(make_padded_string(i)).append(".off"));
      
      std::cout << i+1 << "/" << (opts.nsteps+1) << "\r" << std::flush;
    }
    
  }
  std::cout << "\n";

  return 0;
}

void interpolate(const std::vector<Surface_mesh> &inv_maps, double alpha, double beta, Surface_mesh& result)
{
  //clear output
  result.clear();
  result = inv_maps[0];

  int nv = result.vertices_size();

  for(int j=0; j<nv; ++j){
    Surface_mesh::Vertex v(j);
    // bilinear interpolation
    result.position(v) = bilerp(alpha,beta,inv_maps[0].position(v),inv_maps[1].position(v),inv_maps[2].position(v),inv_maps[3].position(v));
  }
}

void interpolate(const std::vector<Surface_mesh> &inv_maps, double alpha, Surface_mesh& result)
{
  //clear output
  result.clear();
  result = inv_maps[0];

  int nv = result.vertices_size();

  for(int j=0; j<nv; ++j){
    Surface_mesh::Vertex v(j);
    // linear interpolation
    result.position(v) = lerp(alpha,inv_maps[0].position(v),inv_maps[1].position(v));
  }
}

void synthetize_and_save_image(const Surface_mesh& map, const std::string& filename, int res, double expected_mean, bool inv)
{
  MatrixXd img(res,res);
  rasterize_image(map, img);
  img = img * (expected_mean/img.mean());

  if(inv)
    img = 1.-img.array();

  save_image(filename.c_str(), img);
}
