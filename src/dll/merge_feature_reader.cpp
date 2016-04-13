#include "merge_feature_reader.h"
#include <fstream>

//Added by Fuchen Long for the hash pipeline in 12/2/2015
template <typename Dtype>
MergeReader<Dtype>::MergeReader()
{
	Merge_data_ = NULL;
	dim_ = 8192;
}

template <typename Dtype>
MergeReader<Dtype>::~MergeReader()
{
	if (Merge_data_ != NULL)
	{
		delete[]Merge_data_;
		Merge_data_ = NULL;
		std::cout << "release Merge_reader" << std::endl;
	}
}




template class MergeReader<float>;
template class MergeReader<double>;
