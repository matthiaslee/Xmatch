import ctypes
import sys

# load xmatch
libxmatch = ctypes.CDLL('./BoostXmatch/BoostXmatch/libXmatch.so')

def crossmatch(catalogA, catalogB, segsize, outfolder="./results",
               radius=None, zoneheight=None, numGPUs=None, maxout=None,
               overwrite=False, verbosity=9):
    """
    This is a wrapper function for the Xmatch C++/CUDA implementation.

    Arguments
      Required:
        catalogA (str): path to first catalog
        catalogB (str): path to second catalog
        segsize (int): number of objects per segment for tuning of memory
                       utilization. For best performance:
                       (2 * 24bytes * segsize) * 1.15 ~ Total GPU memory

      Optional:
        outfolder (str): path to output folder
        radius (float): matching radius size in arc seconds, defaults to 1.5
        zoneheight (float): height of declination subdivisions (zones) of the
                          Zones Algorithm in arcseconds, defaults to radius
        numGPUs (int): number of GPUs to use
        maxout (int): maximum number of matches, defaults to 2*segsize
        overwrite (bool): if True, then the output directory will be overwritten
        verbosity (int): verbosity between 0[None] and 9[All]


    Example:
    import Xmatch
    Xmatch.crossmatch("/home/madmaze/dr7mode1radec-150901200.bin",
                      "/home/madmaze/dr7mode1radec-150901200.bin",
                      segsize=6*10e6,
                      numGPUs=1)
    """

    args = ["libXmatch.so",
            "-o", outfolder,
            "-n", str(int(segsize))]

    if radius is not None:
        args.extend(["-r", str(radius)])
    if zoneheight is not None:
        args.extend(["-z", str(zoneheight)])
    if numGPUs is not None:
        args.extend(["-t", str(numGPUs)])
    if maxout is not None:
        args.extend(["-m", str(maxout)])
    if verbosity is not None:
        args.extend(["-v", str(verbosity)])
    if overwrite:
        args.append("-x")

    args.append(catalogA)
    args.append(catalogB)

    char_ptr = ctypes.POINTER(ctypes.c_char)
    char_ptr_ptr = ctypes.POINTER(char_ptr)

    libxmatch.main.argtypes = (ctypes.c_int, char_ptr_ptr)

    argc = len(args)
    argv = (char_ptr * (argc + 1))()
    for i, arg in enumerate(args):
        enc_arg = arg.encode('utf-8')
        argv[i] = ctypes.create_string_buffer(enc_arg)

    ret = libxmatch.main(argc, argv)
    if ret != 0:
        raise Exception("Xmatch failed to successfully complete matching")
