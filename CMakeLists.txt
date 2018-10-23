project(otmap)

cmake_minimum_required(VERSION 3.2.0)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set (CMAKE_CXX_STANDARD 11)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(USE_CIMAGE true)

if(USE_CIMAGE)
    add_definitions(-D_cimg_)
    add_definitions(-Dcimg_display=0)

    find_package(PNG)
    if(PNG_FOUND)
        add_definitions(-Dcimg_use_png)
        set(ALLLIBS ${ALLLIBS} png)
    endif()

    find_package(JPEG)
    if(JPEG_FOUND)
        add_definitions(-Dcimg_use_jpeg)
        set(ALLLIBS ${ALLLIBS} jpeg)
    endif()

   if(WIN32)
      add_definitions(-Dcimg_OS=2)
   else()
      add_definitions(-Dcimg_OS=0)
   endif()
endif()

include_directories(extern/)
include_directories(otlib/)

find_package(Cholmod)
if(CHOLMOD_FOUND)
  message(STATUS "Enable Cholmod support")
  add_definitions("-DHAS_CHOLMOD")
  include_directories(${CHOLMOD_INCLUDES})
  set(ALLLIBS ${ALLLIBS} ${CHOLMOD_LIBRARIES})
else()
  message(STATUS "Disable Cholmod support (not found)")
  message(WARNING "Chomod not found, solving will be significantly slower than expected.")
endif()

## FILES ##########################################################################
set(SURFACE_MESH_SRC_FILES
    otlib/surface_mesh/IO.cpp
    otlib/surface_mesh/IO_off.cpp
    otlib/surface_mesh/IO_obj.cpp
    otlib/surface_mesh/Surface_mesh.cpp
)

set(SURFACE_MESH_HEADER_FILES
    otlib/surface_mesh/IO.h
    otlib/surface_mesh/properties.h
    otlib/surface_mesh/Surface_mesh.h
    otlib/surface_mesh/types.h
    otlib/surface_mesh/Vector.h
)

set(OTSOLVER_SRC_FILES
    otlib/transport_map.cpp
    otlib/otsolver_2dgrid.cpp
    otlib/details/line_search.cpp
    otlib/details/nested_dissection.cpp
    otlib/utils/bvh2d.cpp
    otlib/utils/rasterizer.cpp
    otlib/utils/stochastic_rasterizer.cpp
    otlib/utils/mesh_utils.cpp
)

set(OTSOLVER_HEADER_FILES
    otlib/transport_map.h
    otlib/otsolver_2dgrid.h
    otlib/details/line_search.h
    otlib/details/nested_dissection.h
    otlib/utils/bvh2d.h
    otlib/utils/rasterizer.h
    otlib/utils/stochastic_rasterizer.h
    otlib/utils/mesh_utils.h
)

###################################################################################

add_definitions(-DDATA_DIR="${PROJECT_SOURCE_DIR}/data")

if(NOT WIN32)
   add_definitions(-DENABLE_SSE_MODE)
else()
    add_definitions(-D_USE_MATH_DEFINES)
endif()

add_library(otlib ${SURFACE_MESH_SRC_FILES} ${OTSOLVER_SRC_FILES})



set(OTAPPUTILS_SRC_FILES
    apps/common/image_utils.cpp
    apps/common/inputparser.cpp
    apps/common/analytical_functions.cpp
    apps/common/generic_tasks.cpp
)

set(OTAPPUTILS_HEADER_FILES
    apps/common/image_utils.h
    apps/common/inputparser.h
    apps/common/analytical_functions.h
    apps/common/generic_tasks.h
    apps/common/otsolver_options.h
)

add_library(otapputils ${OTAPPUTILS_SRC_FILES})

add_executable(otmap apps/otmap)
target_link_libraries(otmap otapputils otlib ${ALLLIBS})

add_executable(stippling apps/stippling.cpp)
target_link_libraries(stippling otapputils otlib ${ALLLIBS})

add_executable(barycenters apps/barycenters.cpp)
target_link_libraries(barycenters otapputils otlib ${ALLLIBS})
