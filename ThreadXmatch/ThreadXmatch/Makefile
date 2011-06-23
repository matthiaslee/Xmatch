
NVCC	=	nvcc -m 64 -arch sm_20
CCXX	= 	g++ -O2
LD		= 	g++ 
CP		=   cp
RM		=   rm -f

THRUST_DIR = /home/budavari/thrust-v1.3.0

BOOST_DIR = /home/budavari/boost-trunk
BOOST_END = 
BOOST_LIB = \
            -lboost_system${BOOST_END} \
            -lboost_filesystem${BOOST_END} \
            -lboost_date_time${BOOST_END} \
			-lboost_thread${BOOST_END} \
            -lboost_random${BOOST_END} \
            -lboost_program_options${BOOST_END} \

CCFLAGS =	${INCLUDE} 
INCLUDE =	-I${BOOST_DIR} -I${THRUST_DIR} -I/usr/site/cuda/include
LDLIBS  =	-L${BOOST_DIR}/stage/lib ${BOOST_LIB} -lpthread \
			-L/usr/site/cuda/lib64 -lcudart


# --- Targets ---

ThreadXmatch: kernel.o ThreadXmatch.o
	${LD} -o $@ $@.o $< ${LDLIBS}

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
