// This file is part of otmap, an optimal transport solver.
//
// Copyright (C) 2017-2018 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2017 Georges Nader
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <string>
#include <vector>

#include <Eigen/Core>
#include <surface_mesh/Surface_mesh.h>

#include "otsolver_2dgrid.h"
#include "common/analytical_functions.h"
#include "common/inputparser.h"
#include "common/otsolver_options.h"
#include "common/generic_tasks.h"
#include "utils/mesh_utils.h"

using namespace Eigen;
using namespace surface_mesh;
using namespace otmap;

// helps managing CLIoptions ======================================
struct CLIopts : CLI_OTSolverOptions
{
  std::vector<std::string> inputs;
  std::string out_prefix;

  // initializes the options to their default values
  void set_default()
  {
    inputs.clear();

    out_prefix = "";

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

    if(args.getCmdOption("-out", value))
      out_prefix = value[0];

    return true;
  }
};

// ===================================================================
// outputs the usage =================================================
void output_usage()
{
  std::cout << "usage: otmap -in <inputs> <option> <value>" << std::endl;

  std::cout << "input options:" << std::endl;
  std::cout << " * -in input0 [input1] where input* is either a <filename> or a procedural func \":id:res:\"" << std::endl;
  std::cout << " *                     see analytical_functions.h for a list of possible function ids." << std::endl;

  CLI_OTSolverOptions::print_help();

  std::cout << "output options :" << std::endl;
  std::cout << " * -out <prefix>" << std::endl;
}
// ===================================================================


int main(int argc, char**argv)
{
  InputParser args(argc, argv);

  if(args.cmdOptionExists("-help") || args.cmdOptionExists("-h")){
    output_usage();
    return 0;
  }

  CLIopts opts;
  if(!opts.load(args)){
    std::cerr << "invalid input" << std::endl;
    output_usage();
    exit(EXIT_FAILURE);
  }

  std::vector<TransportMap> tmaps;
  generate_transport_maps(opts.inputs, tmaps, opts);


  std::cout << "Save densities, forward and inverse maps...\n";
  for(int k=0; k<tmaps.size(); ++k)
  {
    tmaps[k].fwd_mesh().write(opts.out_prefix + "_" + char('u'+k) + "_fwd.off");

    std::cout << "Transport cost: " << transport_cost(tmaps[k].origin_mesh(), tmaps[k].fwd_mesh(), tmaps[k].density()) << std::endl;

    // compute inverse map
    Surface_mesh inv_map = tmaps[k].origin_mesh();
    apply_inverse_map(tmaps[k], inv_map.points(), opts.verbose_level);
    inv_map.write(opts.out_prefix + "_" + char('u'+k) + "_inv.off");
  }

  std::cout << "Generate and save composite maps...\n";
  if(tmaps.size()==2)
  {
    int img_res = std::sqrt(std::min(tmaps[0].density().size(), tmaps[1].density().size()));
    // compute composite maps u->v and v->u
    Surface_mesh map_uv = tmaps[0].fwd_mesh();
    apply_inverse_map(tmaps[1], map_uv.points(), opts.verbose_level);
    std::cout << "Transport cost of map u->v : " << transport_cost(tmaps[0].origin_mesh(), map_uv, tmaps[0].density()) << std::endl;
    synthetize_and_export_image(map_uv, img_res, tmaps[1].density(), std::string(opts.out_prefix).append("_map_uv_reconstructed"), tmaps[0].density());
    prune_empty_faces(map_uv,tmaps[0].density());
    map_uv.write(std::string(opts.out_prefix).append("_map_uv.off"));

    Surface_mesh map_vu = tmaps[1].fwd_mesh();
    apply_inverse_map(tmaps[0], map_vu.points(), opts.verbose_level);
    std::cout << "Transport cost of map v->u : " << transport_cost(tmaps[1].origin_mesh(), map_vu, tmaps[1].density()) << std::endl;
    synthetize_and_export_image(map_vu, img_res, tmaps[0].density(), std::string(opts.out_prefix).append("_map_vu_reconstructed"), tmaps[1].density());
    prune_empty_faces(map_vu,tmaps[1].density());
    map_vu.write(std::string(opts.out_prefix).append("_map_vu.off"));
  }

  return 0;
}
