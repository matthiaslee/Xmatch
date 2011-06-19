#	PACKAGE:	Xmatch
#
# os: linux
# --- Macros -----------------------------------------------------------------

#CCXX	=	g++ -O2
CCXX	=	g++ -g
LD	=	g++ -static
ARFLAGS	= 	
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

# Compiling C++ code.  The command lines are so long as to be useless.

.cpp.o:
#	@ echo compiling $<
#	@ ${CCXX} -c ${CCFLAGS} $<
	 ${CCXX} -c $< ${CCFLAGS}

.o.x: 
# ${LIBRARY}
#	@ echo linking " " $@
#	@ ${LD} -o $@ ${CCFLAGS} $< ${LDLIBS}
	 ${LD} -o $@ $< ${LDLIBS}






