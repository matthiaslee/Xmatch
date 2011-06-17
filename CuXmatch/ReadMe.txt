
CuXmatch is an 64-bit VS2008 project using CUDA SDK 3.2 (rc?)

Current limitation is that both input and output should fit in host and device memory. Also object lists of {long long, 2xdouble} are sorted by thrust::sort (merging) that needs extra room in the processing step. This translates to about 29M x 29M problems with output  size of 50% more on a GTX 480 with 1.5 GB.



Speed with (zh,sr)=(5,5) is under 10 seconds net time for xmatching :-)