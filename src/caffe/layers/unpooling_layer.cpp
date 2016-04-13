#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

//added by fuchen long in 1/19/2016 
//for the weaklysupervised learning (AutoEncoder)
//And this is the simple upsample, and the max unpooling and the average unpooling is the same
//in this version, the padding is not removal. And to be careful, we should make the kernel and the stride match.

namespace caffe{
	using std::min;
	using std::max;

template<typename Dtype>
void UnPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	PoolingParameter pool_param = this->layer_param_.pooling_param();
	if (pool_param.global_pooling()) {
		CHECK(!(pool_param.has_kernel_size() ||
			pool_param.has_kernel_h() || pool_param.has_kernel_w()))
			<< "With Global_pooling: true Filter size cannot specified";
	}
	else {
		CHECK(!pool_param.has_kernel_size() !=
			!(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
			<< "Filter size is kernel_size OR kernel_h and kernel_w; not both";
		CHECK(pool_param.has_kernel_size() ||
			(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
			<< "For non-square filters both kernel_h and kernel_w are required.";
	}
	CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
		&& pool_param.has_pad_w())
		|| (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
		<< "pad is pad OR pad_h and pad_w are required.";
	CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
		&& pool_param.has_stride_w())
		|| (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
		<< "Stride is stride OR stride_h and stride_w are required.";
	global_pooling_ = pool_param.global_pooling();
	if (global_pooling_) {
		kernel_h_ = bottom[0]->height();
		kernel_w_ = bottom[0]->width();
	}
	else {
		if (pool_param.has_kernel_size()) {
			kernel_h_ = kernel_w_ = pool_param.kernel_size();
		}
		else {
			kernel_h_ = pool_param.kernel_h();
			kernel_w_ = pool_param.kernel_w();
		}
	}
	CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
	CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
	if (!pool_param.has_pad_h()) {
		pad_h_ = pad_w_ = pool_param.pad();
	}
	else {
		pad_h_ = pool_param.pad_h();
		pad_w_ = pool_param.pad_w();
	}
	if (!pool_param.has_stride_h()) {
		stride_h_ = stride_w_ = pool_param.stride();
	}
	else {
		stride_h_ = pool_param.stride_h();
		stride_w_ = pool_param.stride_w();
	}
	if (global_pooling_) {
		CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
			<< "With Global_pooling: true; only pad = 0 and stride = 1";
	}
	if (pad_h_ != 0 || pad_w_ != 0) {
		CHECK(this->layer_param_.pooling_param().pool()
			== PoolingParameter_PoolMethod_AVE
			|| this->layer_param_.pooling_param().pool()
			== PoolingParameter_PoolMethod_MAX)
			<< "Padding implemented only for average and max pooling.";
		CHECK_LT(pad_h_, kernel_h_);
		CHECK_LT(pad_w_, kernel_w_);
	}
}

template<typename Dtype>
void UnPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
		<< "corresponding to (num, channels, height, width)";
	channels_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();
	//get the output size
	unpooled_height_ = static_cast<int>(ceil(static_cast<float>
		((height_ - 1)*stride_h_))) + kernel_h_;
	unpooled_width_ = static_cast<int>(ceil(static_cast<float>
		((width_ - 1)*stride_w_))) + kernel_w_;
	CHECK_EQ(false, pad_h_) <<
		"This Unpooling version doesn't support the padding.";
	CHECK_EQ(false, pad_w_) <<
		"This Unpooling version doesn't support the padding.";
	top[0]->Reshape(bottom[0]->num(), channels_, unpooled_height_,
		unpooled_width_);
	//Fill the mask
	this->FillMask();
}

template<typename Dtype>
void UnPoolingLayer<Dtype>::FillMask(){
	mask_.Reshape(1, 1, unpooled_height_, unpooled_width_);
	int* mask = mask_.mutable_cpu_data();
	//mask_ recoder counts of contributions to each unpooled position
	caffe_set(mask_.count(), 0, mask);
	for (int h = 0; h < height_; ++h){
		for (int w = 0; w < width_; ++w){
			int uhstart = h*stride_h_;
			int uwstart = w*stride_w_;
			int uhend = uhstart + kernel_h_;
			int uwend = uwstart + kernel_w_;
			const int index = h*width_ + w;
			//To filler the output layer
			for (int uh = uhstart; uh < uhend; ++uh){
				for (int uw = uwstart; uw < uwend; ++uw){
					const int unpooled_index = uh*unpooled_width_ + uw;
					mask[unpooled_index] += 1;
				}
			}
		}
	}

}

template<typename Dtype>
void UnPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	const Dtype *bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const int top_count = top[0]->count();
	const int * mask = mask_.cpu_data();
	//in the unpooling there is no mask
	//and at the same time, the average unpool and the max unpool is the same
	//Initialize if use the min, set it FLT_MAX, if not, then set it to zero
	switch (this->layer_param_.pooling_param().pool()){
	case PoolingParameter_PoolMethod_MAX:
		caffe_set(top_count, Dtype(FLT_MAX), top_data);
		//The main loop
		for (int n = 0; n < bottom[0]->num(); ++n){
			for (int c = 0; c < channels_; ++c){
				for (int h = 0; h < height_; ++h){
					for (int w = 0; w < width_; ++w){
						int uhstart = h*stride_h_;
						int uwstart = w*stride_w_;
						int uhend = uhstart + kernel_h_;
						int uwend = uwstart + kernel_w_;
						const int index = h*width_ + w;
						//To filler the output layer
						for (int uh = uhstart; uh < uhend; ++uh){
							for (int uw = uwstart; uw < uwend; ++uw){
								const int unpooled_index = uh*unpooled_width_ + uw;
								// the reverse of the max and become the min
								if (bottom_data[index] < top_data[unpooled_index])
									top_data[unpooled_index] = bottom_data[index];
							}
						}
					}
				}
				//change the channel
				bottom_data += bottom[0]->offset(0, 1);
				top_data += top[0]->offset(0, 1);
			}
		}
		break;
	case PoolingParameter_PoolMethod_AVE:
		caffe_set(top_count, Dtype(0), top_data);
		for (int n = 0; n < bottom[0]->num(); ++n){
			for (int c = 0; c < channels_; ++c){
				for (int h = 0; h < height_; ++h){
					for (int w = 0; w < width_; ++w){
						int uhstart = h*stride_h_;
						int uwstart = w*stride_w_;
						int uhend = uhstart + kernel_h_;
						int uwend = uwstart + kernel_w_;
						const int index = h*width_ + w;
						Dtype data = bottom_data[index];
						//To filler the output layer
						for (int uh = uhstart; uh < uhend; ++uh){
							for (int uw = uwstart; uw < uwend; ++uw){
								const int unpooled_index = uh*unpooled_width_ + uw;
								// to average the output 
								CHECK_GT(mask[unpooled_index], 0);
								top_data[unpooled_index] += data / mask[unpooled_index];
							}
						}
					}
				}
				//change the channel
				bottom_data += bottom[0]->offset(0, 1);
				top_data += top[0]->offset(0, 1);
			}
		}
		break;
	default:
		LOG(FATAL) << "Unknown unpooling method.";
	}
}


	
template <typename Dtype>
void UnPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	if (!propagate_down[0]){
		return;
	}
	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const int * mask = mask_.cpu_data();
	//The back of the unpooling I use the average the residual value
	//Initialize
	caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
	switch (this->layer_param_.pooling_param().pool()){
	case PoolingParameter_PoolMethod_MAX:
		//The main loop
		for (int n = 0; n < top[0]->num(); ++n){
			for (int c = 0; c < channels_; ++c){
				for (int h = 0; h < height_; ++h){
					for (int w = 0; w < width_; ++w){
						int uhstart = h*stride_h_;
						int uwstart = w*stride_w_;
						int uhend = uhstart + kernel_h_;
						int uwend = uwstart + kernel_w_;
						int unpool_size = (uhend - uhstart) * (uwend - uwstart);
						const int index = h*width_ + w;
						//To average the residual value from the top diff
						for (int uh = uhstart; uh < uhend; ++uh){
							for (int uw = uwstart; uw < uwend; ++uw){
								const int unpooled_index = uh*unpooled_width_ + uw;
								bottom_diff[index] += 
									top_diff[unpooled_index] / unpool_size;
							}
						}
					}
				}
				//change the channel
				bottom_diff += bottom[0]->offset(0, 1);
				top_diff += top[0]->offset(0, 1);

			}
		}
		break;
	case PoolingParameter_PoolMethod_AVE:
		//The main loop
		for (int n = 0; n < top[0]->num(); ++n){
			for (int c = 0; c < channels_; ++c){
				for (int h = 0; h < height_; ++h){
					for (int w = 0; w < width_; ++w){
						int uhstart = h*stride_h_;
						int uwstart = w*stride_w_;
						int uhend = uhstart + kernel_h_;
						int uwend = uwstart + kernel_w_;
						int unpool_size = (uhend - uhstart) * (uwend - uwstart);
						const int index = h*width_ + w;
						//To average the residual value from the top diff
						for (int uh = uhstart; uh < uhend; ++uh){
							for (int uw = uwstart; uw < uwend; ++uw){
								const int unpooled_index = uh*unpooled_width_ + uw;
								bottom_diff[index] += 
									top_diff[unpooled_index] / mask[unpooled_index];
							}
						}
					}
				}
				//change the channel
				bottom_diff += bottom[0]->offset(0, 1);
				top_diff += top[0]->offset(0, 1);
			}
		}
		break;
	default:
		LOG(FATAL) << "Unknown unpooling method.";
	}
}

#ifdef CPU_ONLY
STUB_GPU(UnPoolingLayer);
#endif

INSTANTIATE_CLASS(UnPoolingLayer);
REGISTER_LAYER_CLASS(UnPooling);

}// namespace caffe