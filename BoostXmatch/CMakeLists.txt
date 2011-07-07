#The name of the project
PROJECT(BoostXmatch)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

#Searching CUDA and BOOST
FIND_PACKAGE(CUDA 4.0)

# SET BOOST_ROOT=C:\Tamas\Projects\boost-trunk
# SET BOOST_LIBRARYDIR=%BOOST_ROOT%\stage\lib
# SET PATH=%PATH%;%BOOST_ROOT%;%BOOST_LIBRARYDIR%


          set(Boost_USE_STATIC_LIBS        ON)
          set(Boost_USE_MULTITHREADED      ON)
          set(Boost_USE_STATIC_RUNTIME    OFF)

# search for Boost 
FIND_PACKAGE(Boost 1.46 COMPONENTS thread date_time filesystem program_options )


          if(Boost_FOUND)
             include_directories(${Boost_INCLUDE_DIRS})
             add_executable(foo foo.cc)
             target_link_libraries(foo ${Boost_LIBRARIES})
          endif()


#Adding src directory to the project
ADD_SUBDIRECTORY(BoostXmatch)