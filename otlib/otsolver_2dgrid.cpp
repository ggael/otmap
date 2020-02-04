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

#include "otsolver_2dgrid.h"
#include "details/nested_dissection.h"
#include "utils/mesh_utils.h"
#include "utils/BenchTimer.h"
#include <Eigen/Eigenvalues>

using namespace Eigen;
using namespace surface_mesh;

namespace otmap {

GridBasedTransportSolver::
GridBasedTransportSolver()
  : m_verbose_level(1)
{
  #ifdef ENABLE_SSE_MODE
    // set DAZ (denormal as zero) and FTZ (flush-to-zero)
    int oldMXCSR = _mm_getcsr(); /* read the old MXCSR setting */
    int newMXCSR = oldMXCSR | 0x8040; /* set DAZ and FZ bits */
    _mm_setcsr( newMXCSR );
  #endif
}

GridBasedTransportSolver::
~GridBasedTransportSolver()
{
}

void
GridBasedTransportSolver::
adjust_density(VectorXd& density, double max_ratio)
{
  assert(density.size() == m_pb_size);

  // normalise the target so that the integral of density is 1
  double I = density.sum()*m_element_area;
  density /= I;

  // adjust target ratio
  double density_ratio = std::min(max_ratio,
                                 density.maxCoeff()/density.minCoeff());
  double beta = std::max(0. ,
                        (density_ratio*density.minCoeff() - density.maxCoeff())/(1.-density_ratio));

  density = (density.array() + beta) / (1.+beta);

  if(m_verbose_level >= 2){
    std::cout << "[density]" << std::endl;
    std::cout << "  - density integral : " << density.sum()*m_element_area << std::endl;
    std::cout << "  - density stats : min = " << density.minCoeff()
                               << ", max = " << density.maxCoeff()
                               << ", ratio = " << density.maxCoeff() / density.minCoeff()
                               << std::endl;
  }
}

void
GridBasedTransportSolver::
init(int n)
{
  if(m_gridSize==n)
  {
    // we're already all set.
    return;
  }
  if(m_verbose_level>=1)
    std::cout << "Init solver...\n";
  
  BenchTimer timer;
  timer.start();

  m_gridSize = n;
  m_pb_size = n*n;
  
  if(m_mesh!=nullptr) m_mesh.reset(new Surface_mesh);
  else                m_mesh = std::make_shared<Surface_mesh>();
  generate_quad_mesh(n+1, n+1, *m_mesh);

  m_element_area = 1.0/(double(n)*double(n));
  
  this->initialize_laplacian_solver();
  timer.stop();

  if(m_verbose_level>=1)
    std::cout << " - done in " << timer.value(REAL_TIMER) << " s\n";
}

TransportMap
GridBasedTransportSolver::solve(ConstRefVector in_density, SolverOptions opt)
{
  if(m_verbose_level>=1)
  {
    std::cout << " Solve transport map using beta=";
    if(opt.beta==BetaOpt::Zero)               std::cout << "0";
    if(opt.beta==BetaOpt::ConjugateJacobian)  std::cout << "Conjugate-Jacobian";
    std::cout << " ;  max_iter=" << opt.max_iter;
    std::cout << " ;  threshold=" << opt.threshold;
  }

  int n = pb_size();

  // prepare target density
  std::shared_ptr<VectorXd> p_density = std::make_shared<VectorXd>(in_density);
  adjust_density(*p_density, opt.max_ratio);
  m_input_density = p_density.get();

  // initialize cache of vertex gradient at psi=0
  m_cache_1D_g0.setZero(m_mesh->vertices_size(), 2);

  // current and next solution
  VectorXd xk   = VectorXd::Zero(n);
  VectorXd xkp1 = VectorXd::Zero(n);

  // residuals
  VectorXd rkm1  = VectorXd::Zero(n);
  VectorXd rk    = VectorXd::Zero(n);
  VectorXd rkp1  = VectorXd::Zero(n);

  // initial and optimized search directions
  VectorXd d_hat = VectorXd::Zero(n);
  VectorXd d     = VectorXd::Zero(n);

  // update parameter
  double beta = 0;
  double alpha = 0;

  // init
  double residual = compute_residual(xk,rkp1);

  if(m_verbose_level>=1) {
    std::cout << "  ; initial L2=" << residual
              << " Linf=" << rkp1.array().maxCoeff()/m_element_area << "\n";
  }

  double t_linearsolve, t_beta, t_linesearch;
  int it = 0;

  BenchTimer timer;

  double t_linearsolve_sum = 0., t_beta_sum = 0., t_linesearch_sum = 0.;

  while(it < opt.max_iter && residual > opt.threshold){

    if(m_verbose_level>=4) std::cout << " ===> Iteration #" << it+1 << " <===" << std::endl;

    // Find search direction:
    timer.start();

    rkm1.swap(rk);  // same as rkm1 = rk but faster
    rk.swap(rkp1);  // same as rk = rkp1 but faster

    //------------------------------------------------------------
    // Algo 1 - Step 1 - Initial search direction (sec. 4.1)
    //------------------------------------------------------------

    // remember we factorize -L, so no need to negate the result
    d_hat = m_laplacian_solver.solve(rk);

    // make sure the search direction is orthogonal to [1,1,...,1]
    // this corresponds to an orthogonal projection on the hyperplane of normal [1,1,...,1]
    d_hat.array() -= d_hat.mean();

    // check early convergence:
    if(d_hat.norm()<=2*std::numeric_limits<double>::min())
      break;
    
    timer.stop(); t_linearsolve = timer.value(REAL_TIMER); t_linearsolve_sum += t_linearsolve; timer.start();

    //------------------------------------------------------------
    //  Algo 1 - Step 2 - Update search direction (sec. 4.2)
    //------------------------------------------------------------

    if(it<1 || opt.beta==BetaOpt::Zero)
    {
      d = d_hat;
    }
    else // opt.beta==BetaOpt::ConjugateJacobian
    {
      beta = compute_conjugate_jacobian_beta(xk,rkm1,rk,d_hat,d,alpha);

      d = d_hat + beta*d;
    }

    timer.stop(); t_beta = timer.value(REAL_TIMER); t_beta_sum += t_beta; timer.start();

    //------------------------------------------------------------
    //  Algo 1 - Step 3 - Line Search
    //------------------------------------------------------------

    alpha = 0;

    // solve 1D line-search problem using an exact quartic formulation of the error function
    residual = solve_1D_problem(xk, d, rk, residual, /* out */ xkp1, /* out */ rkp1, &alpha);

    // prepare for next iteration:
    xk.swap(xkp1); // same as xk = xkp1 but faster

    timer.stop(); t_linesearch = timer.value(REAL_TIMER); t_linesearch_sum += t_linesearch;
    print_debuginfo_iteration(it, alpha, beta, d, residual, rkp1, t_linearsolve, t_beta, t_linesearch);

    ++it;
  }

  // makes sure m_cache_residual_vtx_grads is uptodate
  compute_vertex_gradients(xk, m_cache_residual_vtx_grads);
  // compute forward mesh
  auto forward_mesh = std::make_shared<Surface_mesh>(*m_mesh);
  for(unsigned int j=0; j<m_cache_residual_vtx_grads.rows(); ++j)
    forward_mesh->points()[j] += m_cache_residual_vtx_grads.row(j).transpose();

  if(m_verbose_level >= 1) {
    std::cout << " Solution:\n";
    std::cout << "  - timings: [" << "solve("
              << t_linearsolve_sum/double(it) << ") + beta("
              << t_beta_sum/double(it) << ") + linesearch("
              << t_linesearch_sum/double(it) << ")] * iters(" << it << ") = " << /*t_linearsolve_sum+t_linesearch_sum+t_beta_sum == */ timer.total(REAL_TIMER) << "s\n";
    std::cout << "  - error L2=" << residual
              <<      "   Linf=" << rkp1.array().maxCoeff()/m_element_area << "\n";
  }
  if(m_verbose_level >= 3) {
    VectorXd ot_cost_per_face;
    compute_transport_cost(m_cache_residual_vtx_grads,ot_cost_per_face);
    std::cout << "  - transport cost=" << ot_cost_per_face.sum() << std::endl;
  }

  return TransportMap(m_mesh, forward_mesh, p_density);
}


double
GridBasedTransportSolver::compute_conjugate_jacobian_beta(ConstRefVector xk, ConstRefVector rkm1, ConstRefVector rk, ConstRefVector d_hat, ConstRefVector d_prev, double alpha) const
{
  int n = pb_size();
  m_cache_beta_Jd.resize(n);
  m_cache_beta_rk_eps.resize(n);

  double eps = alpha/2.;
  compute_residual(xk-eps*d_hat, m_cache_beta_rk_eps);

  m_cache_beta_Jd = (rk-rkm1);
  
  return std::max(-1.,double(m_cache_beta_Jd.dot(m_cache_beta_rk_eps-rk)) / m_cache_beta_Jd.squaredNorm() / (eps) * alpha);
}

//----------------------------------------------------------------

void
GridBasedTransportSolver::
initialize_laplacian_solver()
{
  BenchTimer timer;

  int nv  = m_mesh->vertices_size();
  int nf  = m_mesh->faces_size();

  assert((m_gridSize+1)*(m_gridSize+1)==nv);
  assert(m_gridSize*m_gridSize==nf);

  timer.start();
  {
    // Compute pseudo Laplacian operator on the dual mesh
    typedef Triplet<double,int> Triplet;
    std::vector<Triplet> L_entries;
    L_entries.reserve(nf*5);
    
    // Make L positive by directly forming  -L:
    double w = -0.5;
    // For each face
    for(int i=0; i<m_gridSize; ++i){
      for(int j=0; j<m_gridSize; ++j){
        int id = make_face_index(i,j);
        double sw = 0;
        // Laplacian mask:
        //  2  0  2
        //  0 -8  0  * 1/4
        //  2  0  2 
        // This mask correspond to the used gradient
        int row_id_1 = i == 0 ? i : i-1;
        int col_id_1 = j == 0 ? j : j-1;
        int row_id_2 = i == m_gridSize-1 ? i : i+1;
        int col_id_2 = j == m_gridSize-1 ? j : j+1;

        L_entries.push_back(Triplet(id, make_face_index(row_id_1, col_id_1), w)); sw+=w;
        L_entries.push_back(Triplet(id, make_face_index(row_id_1, col_id_2), w)); sw+=w;
        L_entries.push_back(Triplet(id, make_face_index(row_id_2, col_id_1), w)); sw+=w;
        L_entries.push_back(Triplet(id, make_face_index(row_id_2, col_id_2), w)); sw+=w;
        
        L_entries.push_back(Triplet(id, id, -sw));
      }
    }

    m_mat_L.resize(nf,nf);
    m_mat_L.setFromTriplets(L_entries.begin(), L_entries.end());
  }
  timer.stop();

  if(m_verbose_level>=2) 
  	std::cout << "  - Laplacian matrix computed in " << timer.value(REAL_TIMER) << " s" << std::endl;

  timer.start();
  {
    // weakly enforce psi(0,0)=0
    m_mat_L.coeffRef(0,0) += std::abs(m_mat_L.coeffRef(0,0))*1e4;


  #if HAS_CHOLMOD
    // configure CHOLMOD for best efficiency on our problem
    m_laplacian_solver.setMode(CholmodSupernodalLLt);
    m_laplacian_solver.cholmod().final_asis = 0;
    //m_laplacian_solver.cholmod().final_ll = 1;
    m_laplacian_solver.cholmod().final_resymbol = 1;
    m_laplacian_solver.cholmod().final_super = 0;
    m_laplacian_solver.cholmod().nmethods = 1;
    // m_laplacian_solver.cholmod().method[0].ordering = CHOLMOD_GIVEN;
  #endif

    // Compute custom fill-in permutation
    PermutationMatrix<Dynamic,Dynamic,int> perm(nf);
    nestdiss_ordering(m_gridSize,perm.indices().data());
    m_laplacian_solver.setPermutation(perm);

    m_laplacian_solver.compute(m_mat_L);

    if(m_laplacian_solver.info()!=Success) {
      std::cout << "Solver.Info = ";
      if(m_laplacian_solver.info()==Success) std::cout << "Success\n";
      else if(m_laplacian_solver.info()==NumericalIssue) std::cout << "NumericalIssue\n";
      else if(m_laplacian_solver.info()==NoConvergence) std::cout << "NoConvergence\n";
      else if(m_laplacian_solver.info()==InvalidInput) std::cout << "InvalidInput\n";
      else std::cout << "\n";
    }
    timer.stop();

    if(m_verbose_level>=2) 
      std::cout << "  - Cholesky(Laplacian) done in " << timer.value(REAL_TIMER) << " s\n";
  }

  m_cache_residual_vtx_grads.resize(nv,2);
  m_cache_residual_fwd_area.resize(nf);
  m_cache_beta_Jd.resize(nf);
  m_cache_beta_rk_eps.resize(nf);
}

// Fast version compatible with SIMD
void
GridBasedTransportSolver::
compute_vertex_gradients(ConstRefVector psi, MatrixX2d& vtx_grads) const
{
  unsigned int nv = m_mesh->vertices_size();

  vtx_grads.resize(nv,2);

  using namespace Eigen::internal;
  typedef packet_traits<double>::type Packet;
  const Index PacketSize = packet_traits<double>::size;
  Index simd_size = ((m_gridSize-1)/PacketSize)*PacketSize;

  double w = double(m_gridSize);
  Packet pw05 = pset1<Packet>(0.5*w);
  // inner cells:
  for(Index i=1; i<m_gridSize; ++i){
    int fid0 = make_face_index(i-1,0);
    int fid1 = make_face_index(i,0);
    int vid = make_vtx_index(i,0);

    for(Index j=1; j<simd_size; j+=PacketSize){
      Packet p00 = psi.packet<Unaligned>(fid0+j-1);
      Packet p01 = psi.packet<Unaligned>(fid0+j);
      Packet p10 = psi.packet<Unaligned>(fid1+j-1);
      Packet p11 = psi.packet<Unaligned>(fid1+j);
      vtx_grads.writePacket<Unaligned>(vid+j,0, pmul(pw05,psub(padd(p10,p11),padd(p00,p01))));
      vtx_grads.writePacket<Unaligned>(vid+j,1, pmul(pw05,psub(padd(p01,p11),padd(p00,p10))));
    }

    double p00 = psi(fid0+simd_size-1);
    double p10 = psi(fid1+simd_size-1);
    for(Index j=simd_size; j<m_gridSize; ++j){
      double p01 = psi(fid0+j);
      double p11 = psi(fid1+j);
      vtx_grads(vid+j,0) = 0.5*w*(p10+p11-p00-p01);
      vtx_grads(vid+j,1) = 0.5*w*(p01+p11-p00-p10);
      p00 = p01;
      p10 = p11;
    }
  }

  // boundaries
  for(int k=1; k<m_gridSize; ++k)
  {
    vtx_grads(make_vtx_index(k,0), 0) = w*(psi(make_face_index(k, 0)) - psi(make_face_index(k-1, 0)));
    vtx_grads(make_vtx_index(k,0), 1) = 0.;
    vtx_grads(make_vtx_index(k,m_gridSize), 0) = w*(psi(make_face_index(k, m_gridSize-1)) - psi(make_face_index(k-1, m_gridSize-1)));
    vtx_grads(make_vtx_index(k,m_gridSize), 1) = 0.;

    vtx_grads(make_vtx_index(0,k), 0) = 0.;
    vtx_grads(make_vtx_index(0,k), 1) = w*(psi(make_face_index(0, k)) - psi(make_face_index(0, k-1)));
    vtx_grads(make_vtx_index(m_gridSize,k), 0) = 0;
    vtx_grads(make_vtx_index(m_gridSize,k), 1) = w*(psi(make_face_index(m_gridSize-1, k)) - psi(make_face_index(m_gridSize-1, k-1)));
  }
  // corners
  vtx_grads.row(make_vtx_index(0,0)).setZero();
  vtx_grads.row(make_vtx_index(m_gridSize,0)).setZero();
  vtx_grads.row(make_vtx_index(0,m_gridSize)).setZero();
  vtx_grads.row(make_vtx_index(m_gridSize,m_gridSize)).setZero();
}

EIGEN_DONT_INLINE
void compute_face_area(VectorXd& fwd_area, const MatrixX2d& vtx_grads, int grid_size)
{
  // The following code is SIMD friendly and auto-vectorized by the compiler
  const double e = 1./double(grid_size);
  for(int i=0; i<grid_size; ++i){
    int vid0 = i*(grid_size+1);
    int vid1 = (i+1)*(grid_size+1);
    for(int j=0; j<grid_size; ++j){

      int id = j+i*grid_size;

      int v00 = vid0 + j;
      int v10 = vid1 + j;
      int v01 = vid0 + j+1;
      int v11 = vid1 + j+1;

      // the area of the quad is equal to half the cross product of its diagonal
      fwd_area(id) = 0.5*(  (vtx_grads(v11,0) - vtx_grads(v00,0) + e) * (vtx_grads(v01,1) - vtx_grads(v10,1) + e)
                          - (vtx_grads(v11,1) - vtx_grads(v00,1) + e) * (vtx_grads(v01,0) - vtx_grads(v10,0) - e) );
    }
  }
}

double
GridBasedTransportSolver::
compute_residual(ConstRefVector psi, Ref<VectorXd> out) const
{
  unsigned int nv = m_mesh->vertices_size();

  MatrixX2d &vtx_grads(m_cache_residual_vtx_grads);
  compute_vertex_gradients(psi, vtx_grads);

  VectorXd &fwd_area(m_cache_residual_fwd_area);
  compute_face_area(fwd_area, vtx_grads, m_gridSize);

  out = fwd_area - m_element_area* (*m_input_density);
  double ret = out.squaredNorm() / m_element_area;

  return ret;
}

void
GridBasedTransportSolver::
compute_1D_problem_parameters(Ref<const VectorXd> psi, Ref<const VectorXd> dir, Ref<VectorXd> a, Ref<VectorXd> b) const
{
  // compute a, b, c, such that:
  // r(psi+t*dir) = a*t^2 + b*t + r(psi)
  int nv = m_mesh->vertices_size();
  MatrixX2d &g0(m_cache_1D_g0);
  MatrixX2d &gd(m_cache_1D_gd);
  // No need to recompute the vertex gradient at psi,
  // we already have them from the previous 1D solve.
  // compute_vertex_gradients(psi, g0);
  compute_vertex_gradients(dir, gd);
  
  using namespace Eigen::internal;
  typedef packet_traits<double>::type Packet;
  const Index PacketSize = packet_traits<double>::size;
  Index simd_size = (m_gridSize/PacketSize)*PacketSize;
  Packet p05 = pset1<Packet>(0.5);
  const double e = 1./double(m_gridSize);
  Packet pe  = pset1<Packet>(e);

  for(int i=0; i<m_gridSize; ++i){
    int vid0 = i*(m_gridSize+1);
    int vid1 = (i+1)*(m_gridSize+1);

    for(int j=0; j<simd_size; j+=PacketSize){
      int id = j+i*m_gridSize;
      int v00 = vid0 + j;
      int v10 = vid1 + j;
      int v01 = vid0 + j+1;
      int v11 = vid1 + j+1;

      Packet diag0a_x = padd(psub(g0.packet<Unaligned>(v11,0), g0.packet<Unaligned>(v00,0)), pe);
      Packet diag0a_y = padd(psub(g0.packet<Unaligned>(v11,1), g0.packet<Unaligned>(v00,1)), pe);
      Packet diag0b_x = psub(psub(g0.packet<Unaligned>(v01,0), g0.packet<Unaligned>(v10,0)), pe);
      Packet diag0b_y = padd(psub(g0.packet<Unaligned>(v01,1), g0.packet<Unaligned>(v10,1)), pe);

      Packet dda_x = psub(gd.packet<Unaligned>(v11,0), gd.packet<Unaligned>(v00,0));
      Packet dda_y = psub(gd.packet<Unaligned>(v11,1), gd.packet<Unaligned>(v00,1));
      Packet ddb_x = psub(gd.packet<Unaligned>(v01,0), gd.packet<Unaligned>(v10,0));
      Packet ddb_y = psub(gd.packet<Unaligned>(v01,1), gd.packet<Unaligned>(v10,1));

      // The comment line corresponds to r(psi), but we already have it at hand
      // c.writePacket<Unaligned>(id, p05*( diag0a_x * diag0b_y - diag0a_y * diag0b_x ) - pe*pe*m_input_density->packet<Unaligned>(id));
      a.writePacket<Unaligned>(id, pmul(p05, psub(pmul(dda_x, ddb_y), pmul(dda_y, ddb_x))));
      b.writePacket<Unaligned>(id, pmul(p05, psub(padd(psub(pmul(dda_x, diag0b_y), pmul(dda_y, diag0b_x)), pmul(diag0a_x, ddb_y)), pmul(diag0a_y, ddb_x)) ));
    }

    for(int j=simd_size; j<m_gridSize; ++j){
      int id = j+i*m_gridSize;
      int v00 = vid0 + j;
      int v10 = vid1 + j;
      int v01 = vid0 + j+1;
      int v11 = vid1 + j+1;

      double diag0a_x = g0(v11,0) - g0(v00,0) + e;
      double diag0a_y = g0(v11,1) - g0(v00,1) + e;
      double diag0b_x = g0(v01,0) - g0(v10,0) - e;
      double diag0b_y = g0(v01,1) - g0(v10,1) + e;

      double dda_x = gd(v11,0) - gd(v00,0);
      double dda_y = gd(v11,1) - gd(v00,1);
      double ddb_x = gd(v01,0) - gd(v10,0);
      double ddb_y = gd(v01,1) - gd(v10,1);

      // the comment line corresponds to r(psi), but we already have it at hand
      // c(id) = 0.5*( diag0a_x * diag0b_y - diag0a_y * diag0b_x ) - e*e*(*m_input_density)(id);
      a(id) = 0.5*( dda_x * ddb_y - dda_y * ddb_x);
      b(id) = 0.5*( dda_x * diag0b_y - dda_y * diag0b_x  +  diag0a_x * ddb_y - diag0a_y * ddb_x );
    }
  }
}

double
GridBasedTransportSolver::
solve_1D_problem(ConstRefVector xk, ConstRefVector dir, ConstRefVector rk, double ek, RefVector xk1, RefVector rk1, double *palpha) const
{
  // define aliases
  VectorXd &a(m_cache_1D_a);
  VectorXd &b(m_cache_1D_b);
  a.resize(xk.size());
  b.resize(xk.size());

  compute_1D_problem_parameters(xk, dir, a, b); // ~60%

  // we have: r(xk+t*dir) = rk + a*t^2 + b*t
  // and thus we have: e(xk+t*dir) = r^2 = (1,t,t^2,t^3,t^4) * z
  // with z:
  Matrix<double,5,1> z;
  z <<  ek*m_element_area, // == rk.squaredNorm()
        2.*b.dot(rk),
        b.squaredNorm()+2*a.dot(rk),
        2.*a.dot(b),
        a.squaredNorm(); // ~15%

  // Compute w such that "d e/dt = (1,t,t^2,t^3) * w":
  Matrix<double,4,1> w;
  w << z(1), 2.*z(2), 3.*z(3), 4.*z(4);

  // Find the roots of "d e/dt = (1,t,t^2,t^3) * w" using the companion matrix C
  Matrix3d C(3,3);
  C.setZero();
  C.bottomLeftCorner(2,2).setIdentity();
  C.col(2) = -w.head(3)/w(3);
  EigenSolver<Matrix3d> eig(C);
  // Pick the one being real, close to the [0,1] ideal range, and leading to minimal residual.
  // Note that in most cases there is a single real root, which makes sense as we expect e(t) to be convex.
  double rmin = 0;
  double alpha = 0;
  bool found = false;
  for(int k=0; k<3; ++k)
  {
    std::complex<double> root = eig.eigenvalues()(k);
    if(root.imag()==0. && root.real()>=-10. && root.real()<=10.) {
      double t = root.real();
      // evaluate r^2 at t:
      double r = z(0) + t*(z(1)+t*(z(2)+t*(z(3)+t*z(4))));
      if((!found) || (r<rmin)) {
        found = true;
        alpha = t;
        rmin = r;
      }
    }
  }
  if(!found) {
    alpha = 0.5;
    rmin = z(0) + alpha*(z(1)+alpha*(z(2)+alpha*(z(3)+alpha*z(4))));
  }

  if(palpha)
    *palpha = alpha;
  xk1 = xk + alpha * dir; // ~10%

  // The following three lines are equivalent to compute_residual(xk1, rk1)
  // but exploiting the 1D formulation.
  m_cache_1D_g0 += alpha * m_cache_1D_gd; // ~10%
  rk1 = a*(alpha*alpha)+b*alpha+rk;       // ~10%
  return rmin/m_element_area; // == rk1.squaredNorm()/m_element_area;
}

void
GridBasedTransportSolver::
compute_transport_cost(const MatrixX2d& vtx_grads, VectorXd &cost) const
{
  if(cost.size() != m_gridSize*m_gridSize){
    cost.resize(m_gridSize*m_gridSize);
    cost.setZero();
  }

  // For each face
  for(int i=0; i<m_gridSize; ++i){
    for(int j=0; j<m_gridSize; ++j){

      int id = make_face_index(i,j);

      int v1 = make_vtx_index(i,  j);
      int v2 = make_vtx_index(i+1,j);
      int v3 = make_vtx_index(i+1,j+1);
      int v4 = make_vtx_index(i,  j+1);

      double z1 =  sqrt(1./3.)/2.+0.5;
      double z2 = -sqrt(1./3.)/2.+0.5;

      auto dist2 = [&](double u, double v) {
        return  ((1-u) * (1-v) * vtx_grads.row(v1)
              +     u  * (1-v) * vtx_grads.row(v2)
              +     u  *    v  * vtx_grads.row(v3)
              +  (1-u) *    v  * vtx_grads.row(v4)).squaredNorm();
      };
      cost(id) = (*m_input_density)(id) * m_element_area * (dist2(z1,z1) + dist2(z1,z2) + dist2(z2,z2) + dist2(z2,z1)) / 4.;
    }
  }
}

void
GridBasedTransportSolver::print_debuginfo_iteration(int /*it*/, double alpha, double beta, ConstRefVector search_dir,
                               double l2err, ConstRefVector residual,
                               double t_linearsolve, double t_beta, double t_linesearch) const
{
  if(m_verbose_level>=6) {
    std::cout << "    execution time: search direction = " << t_linearsolve << "s,"
              << " beta = " << t_beta << "s,"
              << " line search = " << t_linesearch << "s,"
              << " [sum = " << t_linearsolve+t_beta+t_linesearch << "s]" << std::endl;
  }
  if(m_verbose_level>=5) {
    std::cout << "    alpha=" << alpha;
    std::cout << "  beta=" << beta;
    std::cout << "  norm(d)=" << search_dir.norm();
    std::cout << "  res^2=" << l2err;
    std::cout << "  Linf="  << residual.array().maxCoeff()/m_element_area << std::endl;
  }
  else if(m_verbose_level>=3)
  {
    std::cout << "    L2 residual = " << l2err << std::endl;
  }
}

}
