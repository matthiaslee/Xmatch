#include "Hello.h"
 
Hello::Hello(const thrust::host_vector<Obj>& data)
{
    m_device_data = data;
}


Obj Hello::First()
{
    return m_device_data[0];
}