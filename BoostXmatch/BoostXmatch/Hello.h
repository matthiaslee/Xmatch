#pragma once

#include "Obj.h"

#pragma warning(push)
#pragma warning(disable: 4996)      // Thrust's use of strerror
#pragma warning(disable: 4251)      // STL class exports
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#pragma warning(pop)
  
using namespace xmatch;

class Hello
{
private:
    thrust::device_vector<Obj> m_device_data;
  
public:
    Hello(const thrust::host_vector<Obj>& data);
    Obj First();
};