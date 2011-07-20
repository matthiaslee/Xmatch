# export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/budavari/boost-trunk/stage/lib

NVCC	=	nvcc -m 64 -gencode=arch=compute_20,code=\"sm_20,compute_20\"
#NVCC	=	nvcc -m 64 -arch sm_20
CCXX	= 	g++ -O2
LD		= 	g++
AR		= 	ar
RM		=   rm -f

#THRUST_DIR = /home/budavari/thrust-v1.3.0

BOOST_DIR = /home/budavari/boost-trunk
BOOST_END = 
BOOST_LIB = \
            -lboost_system${BOOST_END} \
            -lboost_filesystem${BOOST_END} \
            -lboost_date_time${BOOST_END} \
			-lboost_thread${BOOST_END} \
            -lboost_program_options${BOOST_END} \

CCFLAGS =	${INCLUDE} 
INCLUDE =	-I/usr/site/cuda/include
LDLIBS  =	-L. -l${LIBNAME} \
			-L${BOOST_DIR}/stage/lib ${BOOST_LIB} -lpthread \
			-L/usr/site/cuda/lib64 -lcudart -lcuda

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

all: BoostXmatch
lib: ${LINKLIB}

BoostXmatch: BoostXmatch.o ${LINKLIB}
	${LD} -o $@ $< ${OBJS} ${LDLIBS}

BoostXmatch_old: BoostXmatch.o ${OBJS}
	${LD} -o $@ $< ${OBJS} ${LDLIBS}

${LINKLIB} : ${LIBOBJS}
	${AR} ${ARFLAGS} $@ $?

clean :
	- ${RM} *.x *.o


# --- Suffix Rules ---

.SUFFIXES:
.SUFFIXES: .x .o .cpp .cu

.cu.o:
	${NVCC} -c $< ${CCFLAGS}

.cpp.o:
	${CCXX} -c $< ${CCFLAGS}

.o.x: 
	${LD} -o $@ ${$@_OBJ} ${LDLIBS}
