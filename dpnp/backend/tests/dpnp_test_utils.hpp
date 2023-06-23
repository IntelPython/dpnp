#include <iostream>
#include <vector>

#include "dpnp_iterator.hpp"

using namespace std;
using dpnpc_it_t = DPNPC_id<size_t>::iterator;
using dpnpc_value_t = dpnpc_it_t::value_type;
using dpnpc_index_t = dpnpc_it_t::size_type;

template <typename _DataType>
vector<_DataType> get_input_data(const vector<dpnpc_index_t> &shape)
{
    const dpnpc_index_t size =
        accumulate(shape.begin(), shape.end(), dpnpc_index_t(1),
                   multiplies<dpnpc_index_t>());

    vector<_DataType> input_data(size);
    iota(input_data.begin(), input_data.end(),
         1); // let's start from 1 to avoid cleaned memory comparison

    return input_data;
}

template <typename _DataType>
_DataType *get_shared_data(const vector<_DataType> &input_data)
{
    const size_t data_size_in_bytes = input_data.size() * sizeof(_DataType);
    _DataType *shared_data =
        reinterpret_cast<_DataType *>(dpnp_memory_alloc_c(data_size_in_bytes));
    copy(input_data.begin(), input_data.end(), shared_data);

    return shared_data;
}
