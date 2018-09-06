// This file is part of otmap, an optimal transport solver.
//
// Copyright (C) 2017-2018 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2017 Georges Nader
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include "otsolver_2dgrid.h"
#include "common/otsolver_options.h"
#include "utils/eigen_addons.h"
#include "common/image_utils.h"
#include "common/generic_tasks.h"
#include "utils/BenchTimer.h"

using namespace Eigen;
using namespace otmap;

void output_usage()
{
  std::cout << "usage : sample <option> <value>" << std::endl;

  std::cout << std::endl;

  std::cout << "input options : " << std::endl;
  std::cout << " * -in <filename> -> input image" << std::endl;

  std::cout << std::endl;

  CLI_OTSolverOptions::print_help();

  std::cout << std::endl;

  std::cout << " * -ores <res1> <res2> <res3> ... -> ouput point resolutions" << std::endl;
  std::cout << " * -ptscale <value>               -> scaling factor to apply to SVG point sizes (default 1)" << std::endl;
  std::cout << " * -pattern <value>               -> pattern = poisson or a .dat file, default is tiling from uniform_pattern_sig2012.dat" << std::endl;
  std::cout << " * -export_maps                   -> write maps as .off files" << std::endl;

  std::cout << std::endl;

  std::cout << "output options :" << std::endl;
  std::cout << " * -out <prefix>" << std::endl;
}

struct CLIopts : CLI_OTSolverOptions
{
  std::string filename;

  VectorXi ores;
  double pt_scale;
  std::string pattern;
  bool inv_mode;
  bool export_maps;

  std::string out_prefix;

  void set_default()
  {
    filename = "";

    ores.resize(1); ores.setZero();
    ores(0) = 250;

    out_prefix = "";

    pt_scale = 1;
    export_maps = 0;
    pattern = DATA_DIR"/uniform_pattern_sig2012.dat";

    CLI_OTSolverOptions::set_default();
  }

  bool load(const InputParser &args)
  {
    set_default();

    CLI_OTSolverOptions::load(args);

    std::vector<std::string> value;

    if(args.getCmdOption("-in", value))
      filename = value[0];
    else
      return false;

    if(args.getCmdOption("-ores", value)){
      ores.resize(value.size());
      ores.setZero();
      for(unsigned int i=0; i<value.size(); ++i)
        ores(i) = std::atoi(value[i].c_str());
    }

    if(args.getCmdOption("-out", value))
      out_prefix = value[0];

    if(args.getCmdOption("-ptscale", value))
      pt_scale = std::atof(value[0].c_str());

    if(args.getCmdOption("-pattern", value))
      pattern = value[0];

    if(args.cmdOptionExists("-inv"))
      inv_mode = true;
    
    if(args.cmdOptionExists("-export_maps"))
      export_maps = true;

    return true;
  }
};


int main(int argc, char** argv)
{
  setlocale(LC_ALL,"C");

  InputParser input(argc, argv);

  if(input.cmdOptionExists("-help") || input.cmdOptionExists("-h")){
    output_usage();
    return 0;
  }

  CLIopts opts;
  if(!opts.load(input)){
    std::cerr << "invalid input" << std::endl;
    output_usage();
    return EXIT_FAILURE;
  }

  std::vector<Eigen::Vector2d> tile;
  if(!load_point_cloud_dat(opts.pattern, tile))
  {
    std::cerr << "Error loading tile \"" << opts.pattern << "\"\n";
    return EXIT_FAILURE;
  }

  GridBasedTransportSolver otsolver;
  otsolver.set_verbose_level(opts.verbose_level-1);

  if(opts.verbose_level>=1)
    std::cout << "Generate transport map...\n";

  MatrixXd density;
  if(!load_input_density(opts.filename, density))
  {
    std::cout << "Failed to load input \"" << opts.filename << "\" -> abort.";
    exit(EXIT_FAILURE);
  }

  if(density.maxCoeff()>1.)
    density = density / density.maxCoeff(); //normalize

  if(!opts.inv_mode)
    density = 1. - density.array();

  save_image((opts.out_prefix + "_target.png").c_str(), 1.-density.array());


  BenchTimer t_solver_init, t_solver_compute, t_generate_uniform, t_inverse;

  t_solver_init.start();
  otsolver.init(density.rows());
  t_solver_init.stop();

  t_solver_compute.start();
  TransportMap tmap = otsolver.solve(vec(density), opts.solver_opt);
  t_solver_compute.stop();

  std::cout << "STATS solver -- init: " << t_solver_init.value(REAL_TIMER) << "s  solve: " << t_solver_compute.value(REAL_TIMER) << "s\n";

  if(opts.export_maps)
    tmap.fwd_mesh().write(opts.out_prefix + "_fwd.off");

  for(unsigned int i=0; i<opts.ores.size(); ++i){
    
    std::vector<Eigen::Vector2d> points;
    t_generate_uniform.start();
    generate_blue_noise_tile(opts.ores[i], points, tile);
    t_generate_uniform.stop();

    t_inverse.start();
    apply_inverse_map(tmap, points, opts.verbose_level-1);

    // prune outliers
    int c = 0;
    for(int k=0; k<points.size(); ++k)
    {
      Array2i ij = (points[k].array() * Array2d(density.rows(),density.cols())).floor().cast<int>();
      if(density(ij.y(),ij.x())!=0)
      {
        points[c] = points[k];
        ++c;
      }
    }
    points.resize(c);
    t_inverse.stop();

    std::string filename = opts.out_prefix + "_" + std::to_string(opts.ores[i]);
    save_point_cloud_dat(filename + ".dat", points);
    save_point_cloud_eps(filename + ".eps", points, opts.pt_scale);

    std::cout << " # " << opts.ores[i] << "/" << points.size()
                << "  ;  gen: " << t_generate_uniform.value(REAL_TIMER)
                << "s  ;  bvh+inverse: " << t_inverse.value(REAL_TIMER) << "s\n";
  }
}
