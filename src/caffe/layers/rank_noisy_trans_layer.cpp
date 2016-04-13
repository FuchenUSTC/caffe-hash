#include<vector>
#include<iostream>
#include<fstream>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe{
template <typename Dtype>
void RankNoisyTransLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	const int num_output = this->layer_param_.rank_noisy_trans_param().num_output();
	N_ = num_output;
	const int axis = bottom[0]->CanonicalAxisIndex(
		this->layer_param_.rank_noisy_trans_param().axis());
	// To initialize the weight_buff
	// Dimensions starting from "axis" are "flattened" into a single
	// length K_ vector. For example, if bottom[0]'s shape is (N,C,H,W),
	// and axis == 1, N inner product with dimension CHW are performed.
	K_ = bottom[0]->count(axis);
	M_ = bottom[0]->count(0, axis);
	if (this->blobs_.size() > 0){
		LOG(INFO) << "Skipping parameter initialization";
	}
	else{
		this->blobs_.resize(2);
		//Intialize the weight
		vector<int> weight_shape(2);
		weight_shape[0] = N_;
		weight_shape[1] = K_;
		this->blobs_[0].reset(new Blob<Dtype>(weight_shape)); // For the high noisy trans
		this->blobs_[1].reset(new Blob<Dtype>(weight_shape)); // For the low noisy trans
		//Filler
		shared_ptr<Filler<Dtype>> weight_filler(GetFiller<Dtype>(
			this->layer_param_.rank_noisy_trans_param().weight_filler()));
		weight_filler->Fill(this->blobs_[0].get());
		weight_filler->Fill(this->blobs_[1].get());
	}
	this->param_propagate_down_.resize(this->blobs_.size(), true);
	//Reshape the variance and the pow_data
	variance_.Reshape(bottom[0]->num(), 1, 1, 1);
	max_prob_label.Reshape(bottom[0]->num(), 1, 1, 1);
	pow_data_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
	//Reshape the top_diff
	top_diff_high.Reshape(M_, N_, 1, 1);
	top_diff_low.Reshape(M_, N_, 1, 1);
	//Reshape the bottom_data
	bottom_data_high.Reshape(M_, K_, 1, 1);
	bottom_data_low.Reshape(M_, K_, 1, 1);
}

template <typename Dtype>
void RankNoisyTransLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	// Figure out the dimensions
	const int axis = bottom[0]->CanonicalAxisIndex(
		this->layer_param_.rank_noisy_trans_param().axis());
	const int new_K = bottom[0]->count(axis);
	CHECK_EQ(K_, new_K)
		<< "Input size incampatible with inner product parameter.";
	// The first "axis" dimensions are independent inner product;
	// The total number of these is M_, the product over these dimensions.
	M_ = bottom[0]->count(0, axis);
	// The top shape will be the bottom shape with the flattened axes dropped,
	// and replaced by a single axis with dimension num_output (N_).
	vector<int> top_shape = bottom[0]->shape();
	top_shape.resize(axis + 1);
	top_shape[axis] = N_;
	top[0]->Reshape(top_shape);
	top[1]->Reshape(M_, 1, 1, 1);// The pseudo label
}

template <typename Dtype>
void RankNoisyTransLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* bottom_label = bottom[1]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	Dtype* top_label = top[1]->mutable_cpu_data();
	const Dtype* weight_low = this->blobs_[0]->cpu_data();
	const Dtype* weight_high = this->blobs_[1]->cpu_data();
	const int batch_size = bottom[0]->num();
	const int dim = bottom[0]->count() / batch_size;
	get_Variance(bottom_data,batch_size,dim); // compute the variance
	get_Max_label(bottom_data, top_label, bottom_label, batch_size, dim); // find the pseudo label
	//std::ofstream test_out("D:\\35-WeeklySupervisedLearning\\CIFAR10\\Network\\ICLR2015\\Network_train_debug\\test_identity\\rank_noisy_prob.txt", ios::out);
	//test
	//for (int i = 0; i < N_; ++i){
	//	for (int j = 0; j < K_; ++j){
	//		std::cout << weight_high[i*K_ + j] << " ";
	//	}
	//	std::cout << endl;
	//}
	//std::system("pause");
	//
	for (int num = 0; num < batch_size; ++num){
		// The high ranking
		if (top[1]->cpu_data()[num] < 10){
			caffe_cpu_gemv(CblasNoTrans, N_, K_, Dtype(1.0), weight_high, 
				bottom_data + num*dim, Dtype(0.0), top_data + num*dim);
			//for (int i = 0; i < dim; ++i){
			//	std::cout << top_data[num*dim + i] << " ";
			//}
			//std::cout << endl;
			//for (int i = 0; i < dim; ++i){
			//	std::cout << bottom_data[num*dim + i] << " ";
			//}
			//std::cout << endl;
		}
		// The low ranking
		else{
			caffe_cpu_gemv(CblasNoTrans, N_, K_, Dtype(1.0), weight_low,
				bottom_data + num*dim, Dtype(0.0), top_data + num*dim);
		}
	}
	//test_out.close();
}


template <typename Dtype>
void RankNoisyTransLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>*>& bottom){
	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* top_label = top[1]->cpu_data();
	int batch_size = bottom[0]->num();
	const int N_c = N_;
	const int K_c = K_;
	int M_high = 0;
	int M_low = 0;
	// set the top_diff and bottom_data
	for (int num = 0; num < batch_size; ++num){
		// high rank
		if (top_label[num]<10){
			caffe_copy(N_c, top_diff+ num*N_, top_diff_high.mutable_cpu_data() + M_high*N_c);
			caffe_copy(K_c, bottom_data+ num*K_c, bottom_data_high.mutable_cpu_data() + M_high*K_c);
			++M_high;
		}
		// low rank
		else{
			caffe_copy(N_c, top_diff + num*N_c, top_diff_low.mutable_cpu_data() + M_low*N_c);
			caffe_copy(K_c, bottom_data + num*K_c, bottom_data_low.mutable_cpu_data() + M_low*K_c);
			++M_low;
		}
	}
	// Change the weight for BP
	if (this->param_propagate_down_[0]){
		// high rank
		caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_high, (Dtype)1.,
			top_diff_high.cpu_data(), bottom_data_high.cpu_data(), (Dtype)1., this->blobs_[1]->mutable_cpu_diff());
		// low rank
		caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_low, (Dtype)1.,
			top_diff_low.cpu_data(), bottom_data_low.cpu_data(), (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
	}
	if (propagate_down[1]){
		LOG(FATAL) << this->type()
			<< "Layer cannot backpropagate to label inputs";
	}
	if (propagate_down[0]){
		// Gradient with respect to bottom data
		// high rank
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_high, K_, N_, (Dtype)1.,
			top_diff_high.cpu_data(), this->blobs_[1]->cpu_data(), (Dtype)0.,
			bottom_data_high.mutable_cpu_data());
		// low rank
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_low, K_, N_, (Dtype)1.,
			top_diff_low.cpu_data(), this->blobs_[0]->cpu_data(), (Dtype)0.,
			bottom_data_low.mutable_cpu_data());
		M_high = 0;
		M_low = 0;
		// set the bottom data
		for (int num = 0; num < batch_size; ++num){
			// high rank
			if (top_label[num]<10){
				caffe_copy(K_c, bottom_data_high.cpu_data() + M_high*K_c, bottom[0]->mutable_cpu_diff() + num*K_c);
				++M_high;
			}
			// low rank
			else{
				caffe_copy(K_c, bottom_data_low.cpu_data() + M_low*K_c, bottom[0]->mutable_cpu_diff() + num*K_c);
				++M_low;
			}
		}
	}
}

template <typename Dtype>
void RankNoisyTransLayer<Dtype>::get_Variance(const Dtype* bottom_data,
	const int batch_size,const int dim){
	const Dtype pow_num = 2.0;
	caffe_powx(batch_size*dim, bottom_data, pow_num, pow_data_.mutable_cpu_data());
	for (int num = 0; num < batch_size; ++num){
		Dtype ave_data= caffe_cpu_asum(dim, bottom_data + num*dim)/dim;
		Dtype ave_pow_data = caffe_cpu_asum(dim, pow_data_.cpu_data() + num*dim) / dim;
		variance_.mutable_cpu_data()[num] = ave_pow_data - ave_data*ave_data;
	}
}

template<typename Dtype>
void RankNoisyTransLayer<Dtype>::get_Max_label(const Dtype*bottom_data,
	 Dtype* top_label, const Dtype* bottom_label, const int batch_size,
	const int dim){
	Dtype var_margin = this->layer_param_.rank_noisy_trans_param().var_margin_();
	for (int i = 0; i < batch_size; ++i){
		int noisy_label = static_cast<int>(bottom_label[i]);
		float max_prob = bottom_data[i*dim];
		int max_index = 0;
		for (int j = 0; j < dim; ++j){
			if (bottom_data[i*dim + j]>max_prob){
				max_prob = bottom_data[i*dim + j];
				max_index = j;
			}
		}
		if (variance_.cpu_data()[i]>var_margin){
			if (noisy_label == max_index){
				top_label[i] = noisy_label;
			}
			else{
				top_label[i] = -1;
			}			
		}
		else{
			top_label[i] = noisy_label + 10;
		}

	}
}

INSTANTIATE_CLASS(RankNoisyTransLayer);
REGISTER_LAYER_CLASS(RankNoisyTrans);

}