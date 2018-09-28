// This file is part of otmap, an optimal transport solver.
//
// Copyright (C) 2017-2018 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2017 Georges Nader
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

#include <vector>
#include <Eigen/Sparse>

#if HAS_CHOLMOD
#include <Eigen/CholmodSupport>
#endif

#include "surface_mesh/Surface_mesh.h"
#include "transport_map.h"

namespace otmap {

enum struct BetaOpt {
  Zero, ConjugateJacobian
};

struct SolverOptions
{
  BetaOpt beta = BetaOpt::ConjugateJacobian;
  int max_iter = 1000;
  double threshold = 1e-7;
  double max_ratio = std::numeric_limits<double>::max();
};

class GridBasedTransportSolver
{
public:
  GridBasedTransportSolver();
  ~GridBasedTransportSolver();

  /** adjust amount of debug info sent to std::cout */
  inline void set_verbose_level(int v) { m_verbose_level = v; }

  /** Initializes the solver for the given grid size */
  void init(int n);

  /** solve for the given density */
  TransportMap solve(Eigen::Ref<const Eigen::VectorXd> density, SolverOptions opt = SolverOptions());

protected:

  typedef Eigen::Ref<const Eigen::VectorXd> ConstRefVector;
  typedef Eigen::Ref<Eigen::VectorXd>       RefVector;

  /** Assemble and factorize all operators */
  void initialize_laplacian_solver();
  
  void adjust_density(Eigen::VectorXd& density, double max_ratio);

  /** Computes the gradient of each vertex into vtx_grads using psi */
  void compute_vertex_gradients(ConstRefVector psi, Eigen::MatrixX2d& vtx_grads) const;

  void compute_transport_cost(const Eigen::MatrixX2d& vtx_grads, Eigen::VectorXd& cost) const;

  /** Computes the residual of psi */
  double compute_residual(ConstRefVector psi, Eigen::Ref<Eigen::VectorXd> out) const;

  double compute_conjugate_jacobian_beta(ConstRefVector xk, ConstRefVector rkm1, ConstRefVector rk, ConstRefVector d_hat, ConstRefVector d_prev, double alpha) const;

  void compute_1D_problem_parameters(ConstRefVector psi, ConstRefVector dir, RefVector a, RefVector b) const;

  double solve_1D_problem(ConstRefVector xk, ConstRefVector dir, ConstRefVector rk, double ek, RefVector xk1, RefVector rk1, double *palpha = 0) const;

  void print_debuginfo_iteration(int it, double alpha, double beta, ConstRefVector search_dir,
                                 double l2err, ConstRefVector residual,
                                 double t_linearsolve, double t_beta, double t_linesearch) const;

  /** helper functions */
  inline int pb_size() const { return m_pb_size; }
  inline int make_face_index(int i, int j) const { return j+i*m_gridSize     ; }
  inline int make_vtx_index (int i, int j) const { return j+i*(m_gridSize+1) ; }

protected:
  // the working quad mesh
  std::shared_ptr<surface_mesh::Surface_mesh> m_mesh;

  // the input density
  const Eigen::VectorXd* m_input_density;

  double m_element_area; // the initial area of the elements
  int m_gridSize;
  int m_pb_size;

  // the pseudo-Laplacian matrix passed to the solver
  Eigen::SparseMatrix<double> m_mat_L;

  // precomputed Cholesky factorization of L
#if HAS_CHOLMOD
  typedef Eigen::CholmodDecomposition< Eigen::SparseMatrix<double> > LaplacianSolver;
#else
   typedef Eigen::SimplicialLDLT< Eigen::SparseMatrix<double> > LaplacianSolver;
#endif
  LaplacianSolver m_laplacian_solver;

  int m_verbose_level;

  mutable Eigen::MatrixX2d m_cache_residual_vtx_grads;
  mutable Eigen::VectorXd  m_cache_residual_fwd_area;
  mutable Eigen::VectorXd  m_cache_beta_Jd, m_cache_beta_rk_eps;

  mutable Eigen::VectorXd  m_cache_1D_a;
  mutable Eigen::VectorXd  m_cache_1D_b;
  mutable Eigen::MatrixX2d m_cache_1D_g0;
  mutable Eigen::MatrixX2d m_cache_1D_gd;
};

} // namespace otmap
