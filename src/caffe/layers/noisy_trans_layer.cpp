#include <vector>
#include<string>


#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void NoisyTransLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	const int num_output = this->layer_param_.noisy_trans_param().num_output();
	N_ = num_output;
	const int axis = bottom[0]->CanonicalAxisIndex(
		this->layer_param_.noisy_trans_param().axis());
	// To initialize the weight_buff
	// Dimensions starting from "axis" are "flattened" into a single
	// length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
	// and axis == 1, N inner products with dimension CHW are performed.
	K_ = bottom[0]->count(axis);
	if (this->blobs_.size() > 0) {
		LOG(INFO) << "Skipping parameter initialization";
	}
	else
	{
		this->blobs_.resize(1);
		// Intialize the weight
		vector<int> weight_shape(2);
		weight_shape[0] = N_;
		weight_shape[1] = K_;
		this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
		string source = this->layer_param_.noisy_trans_param().source();
		// fill the weights
		if (!source.empty()){
			shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
				this->layer_param_.noisy_trans_param().weight_filler()));
			weight_filler->Fill_noisy(this->blobs_[0].get(), source);
		}
		else{
			shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
				this->layer_param_.noisy_trans_param().weight_filler()));
			weight_filler->Fill(this->blobs_[0].get());
		}
	}
	this->param_propagate_down_.resize(this->blobs_.size(), true);
}


template <typename Dtype>
void NoisyTransLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	// Figure out the dimensions
	const int axis = bottom[0]->CanonicalAxisIndex(
		this->layer_param_.noisy_trans_param().axis());
	const int new_K = bottom[0]->count(axis);
	CHECK_EQ(K_, new_K)
		<< "Input size incompatible with inner product parameters.";
	// The first "axis" dimensions are independent inner products; the total
	// number of these is M_, the product over these dimensions.
	M_ = bottom[0]->count(0, axis);
	// The top shape will be the bottom shape with the flattened axes dropped,
	// and replaced by a single axis with dimension num_output (N_).
	vector<int> top_shape = bottom[0]->shape();
	top_shape.resize(axis + 1);
	top_shape[axis] = N_;
	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void NoisyTransLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const Dtype* weight = this->blobs_[0]->cpu_data();
	weight_num =this-> blobs_[0]->num();
	weight_dim =this-> blobs_[0]->count() / this-> blobs_[0]->num();
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
		bottom_data, weight, (Dtype)0., top_data);

}

template <typename Dtype>
void NoisyTransLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>*>& bottom)
{
	//Change the weight in the backpropagation	
	if (this->param_propagate_down_[0]) {
		const Dtype* top_diff = top[0]->cpu_diff();
		const Dtype* bottom_data = bottom[0]->cpu_data();
	    // Change the weight blobs to the weight_buff value
		/*//Test
		for (int i = 0; i < weight_num; i++)
		{
			for (int j = 0; j < weight_dim; j++)
			{
				std::cout << weight_buff.cpu_data()[i*weight_dim + j] << " ";
			}
			std::cout << endl;
		}
		std::system("pause");
		//*/

		// Gradient with respect to weight
		caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
			top_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_cpu_diff());

	}
	if (propagate_down[0]) {
		const Dtype* top_diff = top[0]->cpu_diff();
		// Gradient with respect to bottom data
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
			top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
			bottom[0]->mutable_cpu_diff());
	}
}

	INSTANTIATE_CLASS(NoisyTransLayer);
	REGISTER_LAYER_CLASS(NoisyTrans);

}