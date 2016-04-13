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
	//bool ReadResizeImage(string image_path);
	//bool ReadResizeImage(
	//	void *scan0,
	//	int width,
	//	int height,
	//	int stride,
	//	int channel);
	//bool ReadResizeImage(cv::Mat& mat);  
	//bool ProcessImage(cv::Mat &cv_img_origin);
	bool ReadTheMergeFeature(float * Merge_fea, int dim);
	inline const Dtype *GetImgData() const { return Merge_data_; }
	inline void SetResize(int resize_width, int resize_height){ resize_width_ = resize_width; resize_height_ = resize_height; }
	inline void SetCropSize(int crop_width, int crop_height)
	{
		crop_width_ = crop_width;
		crop_height_ = crop_height;
		Merge_data_ = new Dtype[3 * crop_width_ * crop_height_];
	}

private:
	Dtype *Merge_data_;
	int dim_;
	//int resize_width_;
	//int resize_height_;
	//int crop_width_;
	//int crop_height_;
};


#endif