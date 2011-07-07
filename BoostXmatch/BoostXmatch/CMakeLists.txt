#Linking with libcutil (expected to be in ${CUDA_SDK_ROOT_DIR}/C/lib)
##LINK_LIBRARIES("-L/${CUDA_SDK_ROOT_DIR}/C/lib -lcutil")

INCLUDE(FindCUDA)
INCLUDE(FindBoost)

#The rule to build the executable program 
CUDA_ADD_EXECUTABLE(BoostXmatch BoostXmatch.cpp Log.cpp CudaManager.cpp Obj.cpp Segment.cpp SegmentManager.cpp Job.cpp JobManager.cpp Sorter.cu Worker.cu )