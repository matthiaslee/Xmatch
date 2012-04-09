NVCC	=	nvcc -m 64 -gencode=arch=compute_20,code=\"sm_20,compute_20\"
CCXX	= 	g++ -O6
LD		= 	g++
AR		= 	ar
RM		=   rm -f

# ----------- 
# either export these variables 
#
#export BOOST_DIR=/path/to/boost
#export CUDA_INCLUDE=/opt/cuda/include
#export CUDA_LIB=/opt/cuda/lib64 
#
# or set them here
#
#CUDA_INCLUDE=
#CUDA_LIB=
#BOOST_DIR=

BOOST_END = 
BOOST_LIB = \
            -lboost_system${BOOST_END} \
            -lboost_filesystem${BOOST_END} \
            -lboost_date_time${BOOST_END} \
			-lboost_thread${BOOST_END} \
            -lboost_program_options${BOOST_END} \

CCFLAGS =	${INCLUDE} 
INCLUDE =	-I${CUDA_INCLUDE} -I${BOOST_DIR}
LDLIBS  =	-L. -l${LIBNAME} \
			-L${BOOST_DIR}/stage/lib ${BOOST_LIB} -lpthread \
			-L${CUDA_LIB} -lcudart -lcuda

ARFLAGS = 	-rcs

LIBNAME = 	xmatch
LIBOBJS = 	Log.o \
			CudaContext.o \
			CudaManager.o \
			Obj.o \
			Segment.o \
			SegmentManager.o \
			Job.o \
			JobManager.o \
			Common.o \
			Sorter.o \
			Worker.o \

LINKLIB = 	lib${LIBNAME}.a
			

# --- Targets ---

all: BoostXmatch.x
lib: ${LINKLIB}

BoostXmatch.x: BoostXmatch.o lib
	${LD} -o $@ $< ${OBJS} ${LDLIBS}

${LINKLIB} : ${LIBOBJS}
	${AR} ${ARFLAGS} $@ $?

clean :
	- ${RM} *.x *.o lib*.a


# --- Suffix Rules ---

.SUFFIXES:
.SUFFIXES: .cpp .cu .o

.cu.o:
	${NVCC} -c $< ${CCFLAGS}

.cpp.o:
	${CCXX} -c $< ${CCFLAGS}

