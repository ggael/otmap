// This file is part of otmap, an optimal transport solver.
//
// Copyright (C) 2017-2018 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <Eigen/Core>

namespace otmap {

typedef std::function<double(Eigen::Ref<const Eigen::VectorXd>,Eigen::Ref<Eigen::VectorXd>)> ResidualFunction;

class QuadraticLineSearch
{
public:
  
  QuadraticLineSearch() { init(); }
  
  QuadraticLineSearch(int pb_size) : m_cache_residual(pb_size) { init(); }

  void pre_allocate(int pb_size) { m_cache_residual.resize(pb_size); }

  void attach_residual_func(const ResidualFunction &func) { m_func = func; }

  /* \param width_tolerance stopping threshold (stop when |alpha_k = alpha_{k-1}| <= width_tolerance)
   * \param low_boundary lower bound for alpha
   * \param high_boundary upper bound for alpha
   */
  void set_tolerance_and_bounds(double width_tolerance, double low_boundary, double high_boundary) {
    m_width_tolerance = width_tolerance;
    m_low_boundary    = low_boundary;
    m_high_boundary   = high_boundary;
  }

  void set_verbose_level(int l) { m_verbose_level = l; }

  /**
    * \param xk starting position
    * \param dir search direction
    * \param[out] xk1 solution (xk1 = xk + alpha * dir)
    * \param[out] rk1 residual vector at xk1
    * \param ek initial error at xk (a negative number means that it must be recomputed)
    * \param[out] palpha return value of alpha
    */
  double operator()(Eigen::Ref<const Eigen::VectorXd> xk, Eigen::Ref<const Eigen::VectorXd> dir,
                    Eigen::Ref<Eigen::VectorXd> xk1, Eigen::Ref<Eigen::VectorXd> rk1,
                      double ek = -1, double *palpha = 0) const;
protected:
  void init();
  double m_width_tolerance, m_low_boundary, m_high_boundary;
  mutable Eigen::VectorXd m_cache_residual;
  ResidualFunction m_func;
  int m_verbose_level;
};

} // namespace otmap
