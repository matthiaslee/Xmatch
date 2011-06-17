
TODO:
- Wrap-around in RA
- Smarter I/O

Known bugs: none

Linux command line to build is something like:
	nvcc -m 64 -arch sm_20 -I CuXmatch.cu

Windows command line to build is more like:
	nvcc -m 64 -arch sm_20 -I "c:\CUDA\include" -L "C:\CUDA\lib\x64" new.cu
	nvcc -m 64 -arch sm_20 -I "c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v3.2\include" -L "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v3.2\lib\x64" CuXmatch.cu
