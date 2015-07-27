cmake_minimum_required(VERSION 2.8)
project( Superpixels )

SET(OpenCV_DIR "~/usr/local/include/opencv")
INCLUDE_DIRECTORIES(
	${OpenCV_DIR} 
	include
)


find_package( OpenCV REQUIRED )
find_package(CUDA QUIET REQUIRED)

cuda_add_executable(
	Sp_demo 
	demo.cpp
	Superpixels.cpp 
	my_sp_struct.h
	RgbLab.h
	RgbLab.cu
	init_seg.h
	init_seg.cu
	sp_helper.h
	sp_helper.cu
	update_param.h
	update_param.cu
	update_seg.h
	update_seg.cu

)
cuda_add_executable(
	Sp_demo_for_direc
	demo_for_direc.cpp
	Superpixels.cpp 
	my_sp_struct.h
	RgbLab.h
	RgbLab.cu
	init_seg.h
	init_seg.cu
	sp_helper.h
	sp_helper.cu
	update_param.h
	update_param.cu
	update_seg.h
	update_seg.cu

)
TARGET_LINK_LIBRARIES(Sp_demo ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(Sp_demo_for_direc ${OpenCV_LIBS})


