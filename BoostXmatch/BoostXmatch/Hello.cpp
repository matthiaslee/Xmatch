/*
 *   ID:          $Id: BoostXmatch.cpp 6993 2011-06-30 04:42:53Z budavari $
 *   Revision:    $Rev: 6993 $
 */
#include "Hello.h"
#include <iostream>

int main(int argc, char* argv[])
{
	 Obj e;
	 e.mId = 99;
	 e.mRa = 2;

	 thrust::host_vector<Obj> host_data(10);
	 host_data[0] = e;

     Hello hello(host_data);
 
     Obj ee = hello.First();
     std::cout << "The 1st is: " << ee.mId << " " << ee.mRa << std::endl;
  
     return 0;
}
