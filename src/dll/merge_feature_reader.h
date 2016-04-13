#ifndef MERGE_FEATURE_READER_H_
#define MERGE_FEATURE_READER_H_

#include <iostream>
#include <string>
#include <opencv\highgui.h>
#include <opencv2/imgproc/imgproc.hpp>
using std::string;

template <typename Dtype>
class MergeReader
{
public:
	MergeReader();
	~MergeReader();

	inline bool ReadTheMergeFeature(float * Merge_fea, int dim)
	{
		Merge_data_ = new Dtype[dim];
		for (int i = 0; i < dim; i++)
		{
			Merge_data_[i] = Merge_fea[i];
		}
		return true;
	}
	inline const Dtype *GetMergeData() const { return Merge_data_; }


private:
	Dtype *Merge_data_;
	int dim_;

};


#endif