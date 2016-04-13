#include <algorithm>
#include <functional>
#include <cfloat>
#include <vector>
#include <iostream>
#include <utility>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

// The author: XiaoTong
// The modified or changed by Fuchen Long
namespace caffe {

	template <typename Dtype>
	void SoftmaxWithNoisyLabelLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		if (this->layer_param_.loss_weight_size() == 0) {
			this->layer_param_.add_loss_weight(Dtype(1));
			for (int i = 1; i < top.size(); ++i) {
				this->layer_param_.add_loss_weight(0);
			}
		}
		LayerParameter softmax_param(this->layer_param_);
		softmax_param.set_type("Softmax");
		softmax_layer_y_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
		softmax_layer_z_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
		softmax_bottom_vec_y_.clear();
		softmax_bottom_vec_z_.clear();
		softmax_top_vec_y_.clear();
		softmax_top_vec_z_.clear();
		softmax_bottom_vec_y_.push_back(bottom[0]);
		softmax_bottom_vec_z_.push_back(bottom[1]);
		softmax_top_vec_y_.push_back(&prob_y_);
		softmax_top_vec_z_.push_back(&prob_z_);
		softmax_layer_y_->SetUp(softmax_bottom_vec_y_, softmax_top_vec_y_);
		softmax_layer_z_->SetUp(softmax_bottom_vec_z_, softmax_top_vec_z_);

		N_ = bottom[0]->channels();
		this->blobs_.resize(1);
		this->blobs_[0].reset(new Blob<Dtype>(N_, N_, 1, 1));

		string source = this->layer_param_.softmax_noisy_label_loss_param().source();
		// fill the weights
		if (!source.empty()){
			shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
				this->layer_param_.softmax_noisy_label_loss_param().matrix_c_filler()));
			weight_filler->Fill_noisy(this->blobs_[0].get(), source);
		}
		else{
			shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
				this->layer_param_.softmax_noisy_label_loss_param().matrix_c_filler()));
			weight_filler->Fill(this->blobs_[0].get());
		}
		
		CHECK_EQ(this->blobs_[0]->num(), N_);
		CHECK_EQ(this->blobs_[0]->channels(), N_);
		CHECK_EQ(this->blobs_[0]->height(), 1);
		CHECK_EQ(this->blobs_[0]->width(), 1);

		lr_z_ = this->layer_param_.softmax_noisy_label_loss_param().update_noise_lr(); //The learning rate 
	}

	template <typename Dtype>
	void SoftmaxWithNoisyLabelLossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::Reshape(bottom, top);
		softmax_layer_y_->Reshape(softmax_bottom_vec_y_, softmax_top_vec_y_);
		softmax_layer_z_->Reshape(softmax_bottom_vec_z_, softmax_top_vec_z_);

		M_ = bottom[0]->num();
		CHECK_EQ(bottom[0]->channels(), N_);
		posterior_.Reshape(M_, N_, NumNoiseType, 1);

		if (top.size() >= 2) top[1]->ReshapeLike(prob_y_);
		if (top.size() >= 3) top[2]->ReshapeLike(prob_z_);
		if (top.size() >= 4) top[3]->ReshapeLike(posterior_);
	}

	template <typename Dtype>
	void SoftmaxWithNoisyLabelLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		// Compute conditional probability of p(y|x) and p(z|x)
		softmax_layer_y_->Forward(softmax_bottom_vec_y_, softmax_top_vec_y_);
		softmax_layer_z_->Forward(softmax_bottom_vec_z_, softmax_top_vec_z_);

		// Compute posterior
		const Dtype* C = this->blobs_[0]->cpu_data();
		const Dtype* noisy_label = bottom[2]->cpu_data();
		const Dtype* p_y_given_x = prob_y_.cpu_data();
		const Dtype* p_z_given_x = prob_z_.cpu_data();
		const Dtype uniform = Dtype(1.0) / (N_ - 1);
		const int dim_yz = N_ * NumNoiseType;
		Dtype* posterior_data = posterior_.mutable_cpu_data();
		Dtype loss = 0;
		for (int i = 0; i < M_; ++i) {
			if (noisy_label[i] == -1) {
				// Unlabeled
				caffe_memset(dim_yz * sizeof(Dtype), 0, posterior_data + i * dim_yz); // for the unlabeled data, to set 0,one question
				continue;                                                               // why there is data unlabeled? why not all the 
			}                                                                         // data should have the label, true or error because 
			// know the true label, or predict the true label.
			for (int y = 0; y < N_; ++y) {										      // the data is the noisy data.
				int offset = posterior_.offset(i, y);
				Dtype py = p_y_given_x[i * N_ + y];
				for (int z = 0; z < NumNoiseType; ++z) {
					Dtype pz = p_z_given_x[i * NumNoiseType + z];
					switch (NoiseType(z)) {
					case NoiseFree:
						posterior_data[offset + z] = (y == noisy_label[i]);   // only let the noisy label i get the posterior
						break;
					case RandomNoise:
						posterior_data[offset + z] = uniform * (y != noisy_label[i]); // like the noisy free condition
						break;
					case ConfusingNoise:
						posterior_data[offset + z] =
							C[static_cast<int>(noisy_label[i] * N_ + y)]; //know the noisy label and find the noisy tarns in the matrix
						break;
					default:
						break;
					}
					posterior_data[offset + z] *= py * pz;
				}
			}
			// Compute loss
			Dtype sum = 0;
			Dtype weighted_loss = 0;
			for (int y = 0; y < N_; ++y) {
				for (int z = 0; z < NumNoiseType; ++z) {          //compute the softmax loss
					Dtype p = posterior_.data_at(i, y, z, 0);
					sum += p;
					weighted_loss -= p * log(std::max(p, Dtype(FLT_MIN)));
				}
			}
			if (sum > 0) {
				loss += weighted_loss / sum;                 // devided the summation of the probability
				caffe_scal(dim_yz, Dtype(1.0) / sum, posterior_data + i * dim_yz); // scale the posterior for the back propagation
			}
		}
		top[0]->mutable_cpu_data()[0] = loss / M_;
		if (top.size() >= 2) top[1]->ShareData(prob_y_);
		if (top.size() >= 3) top[2]->ShareData(prob_z_);
		if (top.size() >= 4) top[3]->ShareData(posterior_);
	}

	template <typename Dtype>
	void SoftmaxWithNoisyLabelLossLayer<Dtype>::Backward_cpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		if (propagate_down[2]) {
			LOG(FATAL) << this->type()
				<< " Layer cannot backpropagate to label inputs.";
		}
		if (propagate_down[0]) { // Back prop: y
			Blob<Dtype> true_prob(M_, N_, 1, 1);
			Dtype* p = true_prob.mutable_cpu_data();
			const Dtype* p_yz = posterior_.cpu_data();
			for (int i = 0; i < M_; ++i) {
				for (int y = 0; y < N_; ++y) {
					int offset = posterior_.offset(i, y);
					Dtype sum = 0;
					for (int z = 0; z < NumNoiseType; ++z) sum += p_yz[offset + z]; // for the row addition
					p[i * N_ + y] = sum;
				}
			}
			BackProp(prob_y_, true_prob, top[0]->cpu_diff()[0], bottom[0]);
		}
		if (propagate_down[1]) { // Back prop: z and the lr_z_ is the adjustion for the noisy type learning
			Blob<Dtype> true_prob(M_, NumNoiseType, 1, 1);
			Dtype* p = true_prob.mutable_cpu_data();
			const Dtype* p_yz = posterior_.cpu_data();
			for (int i = 0; i < M_; ++i) {
				for (int z = 0; z < NumNoiseType; ++z) {
					Dtype sum = 0;
					for (int y = 0; y < N_; ++y)
						sum += p_yz[i * N_ * NumNoiseType + y * NumNoiseType + z];  // for the column addition
					p[i * NumNoiseType + z] = sum;
				}
			}
			BackProp(prob_z_, true_prob, top[0]->cpu_diff()[0] * lr_z_, bottom[1]);
		}
	}

	template <typename Dtype>
	void SoftmaxWithNoisyLabelLossLayer<Dtype>::BackProp(const Blob<Dtype>& prob, // this optimazation is used in the parameter for all the theta not the input pz or py
		const Blob<Dtype>& true_prob, Dtype lr, Blob<Dtype>* diff) {
		const Dtype* prob_data = prob.cpu_data();
		const Dtype* true_prob_data = true_prob.cpu_data();
		Dtype* diff_data = diff->mutable_cpu_diff();
		caffe_sub(diff->count(), prob_data, true_prob_data, diff_data); // The question 2: why substract the true_prob_data?
		// set diff of unlabeled samples to 0
		const int N = prob.channels();
		for (int i = 0; i < M_; ++i) {
			Dtype sum = 0;
			for (int j = 0; j < N; ++j) sum += true_prob_data[i * N + j];
			for (int j = 0; j < N; ++j) diff_data[i * N + j] *= sum;
		}
		caffe_scal(diff->count(), lr / M_, diff_data);
	}

	INSTANTIATE_CLASS(SoftmaxWithNoisyLabelLossLayer);
	REGISTER_LAYER_CLASS(SoftmaxWithNoisyLabelLoss);

}