# PACKAGE:	Xmatch
#
# OS: Linux

#CC	=	g++ -O2
CC	=	g++ -g
LD	=	g++ -static
CP	=       cp
RM	=       rm -f

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
INCLUDE =	-I${BOOST_DIR}

LDLIBS  =	-L${BOOST_DIR}/stage/lib ${BOOST_LIB} -lpthread

EXE =	BoostXmatch.x

# --- Targets ----------------------------------------------------------------

# Rules for building each executable.
all :	${EXE}

# Test
test : ${EXE}
	./${EXE} ../../Data/test.idradec ../../Data/test.idradec -v

# Build and install the executables.
apps :	${EXE}
	- ${CP} ${EXE} ../bin

install: apps

# Clean without removing installed products.
clean :
	- ${RM} ${EXE} *.x *.o


# Remove all files generated or installed by make.  
veryclean :	clean
	- cd ../bin; ${RM} ${EXE}


# --- Suffix Rules -----------------------------------------------------------

# First clear out the default suffixes, then declare our own and
# define the rules.

.SUFFIXES:
.SUFFIXES: .x .o .cpp

# Compiling C++ code

.cpp.o:
	 ${CC} -c $< ${CCFLAGS}

.o.x: 
	 ${LD} -o $@ $< ${LDLIBS}






