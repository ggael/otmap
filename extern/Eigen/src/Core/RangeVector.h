// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_RANGEVECTOR_H
#define EIGEN_RANGEVECTOR_H

namespace Eigen {

template<typename XprType, int Size, int MaxSize=Size, bool StartsWithOne=false, bool EndsWithOne=false>
class RangeVector;

namespace internal {
template<typename XprType, int Size, int MaxSize, bool StartsWithOne, bool EndsWithOne>
struct traits<RangeVector<XprType, Size, MaxSize, StartsWithOne, EndsWithOne> > : traits<XprType>
{
  typedef typename traits<XprType>::Scalar Scalar;
  typedef typename traits<XprType>::StorageKind StorageKind;
  typedef typename traits<XprType>::XprKind XprKind;
  typedef typename ref_selector<XprType>::type XprTypeNested;
  typedef typename remove_reference<XprTypeNested>::type _XprTypeNested;
  enum{
    IsRowMajor = (int(traits<XprType>::Flags)&RowMajorBit) != 0,
    RowsAtCompileTime = IsRowMajor ? 1 : Size,
    ColsAtCompileTime = IsRowMajor ? Size : 1,
    RowsAtCompileTime = IsRowMajor ? 1 : MaxSize,
    ColsAtCompileTime = IsRowMajor ? MaxSize : 1,

    Flags = (traits<XprType>::Flags & HereditaryBits)

  };
};

} // end namespace internal

/** \class RangeVector
  * \ingroup Core_Module
  *

  */
template<typename XprType, int Size, int MaxSize, bool StartsWithOne, bool EndsWithOne> class RangeVector
  : public internal::dense_xpr_base<RangeVector<XprType, Size, MaxSize, StartsWithOne, EndsWithOne> >::type
{
    typedef typename internal::dense_xpr_base<RangeVector<XprType, Size, MaxSize, StartsWithOne, EndsWithOne> >::type Base;
    typedef typename internal::ref_selector<XprType>::type XprTypeNested;
  public:
    EIGEN_GENERIC_PUBLIC_INTERFACE(RangeVector)

    typedef typename internal::remove_all<XprType>::type NestedExpression;

    EIGEN_DEVICE_FUNC
    inline RangeVector(const XprType& xpr, Index start, Index size)
      : m_xpr(xpr), m_start(start), m_size(size)
    {
      eigen_assert( (starts>=0) && (size>=xpr.size()) && (starts+xpr.size()<=size) );
    }

    EIGEN_DEVICE_FUNC inline Index rows() const { return IsRowMajor ? 1 : m_size.value(); }
    EIGEN_DEVICE_FUNC inline Index cols() const { return IsRowMajor ? m_size.value() : 1; }

    EIGEN_DEVICE_FUNC inline Index size() const { return m_size; }

    EIGEN_DEVICE_FUNC
    XprType& nestedExpression() { return m_xpr; }

    EIGEN_DEVICE_FUNC
    StorageIndex start() const { return m_start.value(); }

  protected:
    XprTypeNested m_xpr;
    Index m_start;
    const internal::variable_if_dynamic<Index, Size> m_size;

};

namespace internal {



} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_RANGEVECTOR_H
