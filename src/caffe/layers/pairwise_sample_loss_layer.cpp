#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

namespace caffe{
	template <typename Dtype>
	void PairWiseSampleLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*> &top){
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		// 0: feature 1: probability 2:label
		CHECK_EQ(bottom[0]->num(), bottom[1]->num());
		CHECK_EQ(bottom[1]->num(), bottom[2]->num());
		CHECK_EQ(bottom[0]->width(), 1);
		CHECK_EQ(bottom[0]->height(), 1);
		CHECK_EQ(bottom[1]->width(), 1);
		CHECK_EQ(bottom[1]->height(), 1);
		CHECK_EQ(bottom[2]->width(), 1);
		CHECK_EQ(bottom[2]->height(), 1);

		// vector for pairwise learning
		noisy_label_flag = this->layer_param_.pairwise_sample_loss_param().noisy_flag();
		pairwise_threshold = this->layer_param_.pairwise_sample_loss_param().num();
		p_margin = this->layer_param_.pairwise_sample_loss_param().margin();
		positive_sample.Reshape(pairwise_threshold, bottom[0]->channels(), 1, 1);
		negative_sample.Reshape(pairwise_threshold, bottom[0]->channels(), 1, 1);
		diff_.Reshape(pairwise_threshold, bottom[0]->channels(), 1, 1);
		dist_sq_.Reshape(pairwise_threshold, bottom[0]->channels(), 1, 1);
		positive_class.Reshape(pairwise_threshold, 1, 1, 1);
		negative_class.Reshape(pairwise_threshold, 1, 1, 1);
		positive_index.Reshape(pairwise_threshold, 1, 1, 1);
		negative_index.Reshape(pairwise_threshold, 1, 1, 1);
		alpha.Reshape(pairwise_threshold, 1, 1, 1);
		gradient.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
		pseudo_label_.Reshape(bottom[0]->num(), 1, 1, 1);
		batch_size = bottom[0]->num();
		fea_dim = bottom[0]->channels();
		class_dim = bottom[1]->channels();
	}

	template <typename Dtype>
	void PairWiseSampleLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
		
		if (noisy_label_flag)
			Sample_pair(bottom);
		else
			Sample_pair_pseudo(bottom);
		int p_cla;
		int n_cla;
		int p_ind;
		int n_ind;
		int offset_;
		Dtype loss(0.0);
		const Dtype* prob = bottom[1]->cpu_data();
		for (int i = 0; i < pairwise_num; ++i){
			offset_ = fea_dim*i;
			p_cla = positive_class.cpu_data()[i];
			n_cla = negative_class.cpu_data()[i];
			p_ind = positive_index.cpu_data()[i];
			n_ind = negative_index.cpu_data()[i];
			alpha.mutable_cpu_data()[i] = 1 
				- 0.5*abs(prob[p_ind*class_dim + p_cla] - prob[n_ind*class_dim + p_cla])
				- 0.5*abs(prob[p_ind*class_dim + n_cla] - prob[n_ind*class_dim + n_cla]);
			caffe_sub(fea_dim, positive_sample.cpu_data()+offset_, negative_sample.cpu_data()+offset_, diff_.mutable_cpu_data()+offset_);
			caffe_scal(fea_dim, alpha.cpu_data()[i], diff_.mutable_cpu_data() + offset_);
			caffe_copy(fea_dim, diff_.cpu_data() + offset_, dist_sq_.mutable_cpu_data() + offset_);
			for (int j = 0; j < fea_dim; ++j){
				loss += std::max(p_margin*alpha.cpu_data()[i] - dist_sq_.cpu_data()[offset_ + j], Dtype(0.0));
			}
		}
		loss = loss / static_cast<Dtype>(pairwise_num) /Dtype(2.0);  // the scale 2 to be considered 
		top[0]->mutable_cpu_data()[0] = loss;
	}

	template <typename Dtype>
	void PairWiseSampleLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_dowm, const vector<Blob<Dtype>*>& bottom){
		Dtype* bout = bottom[0]->mutable_cpu_diff();
		caffe_set(fea_dim*batch_size, Dtype(0.0), gradient.mutable_cpu_data());
		int p_ind;
		int n_ind;
		// positive
		for (int i = 0; i < pairwise_num; ++i){
			p_ind = positive_index.cpu_data()[i];
			for (int j = 0; j < fea_dim; ++j){
				if (p_margin*alpha.cpu_data()[i] - dist_sq_.cpu_data()[i*fea_dim + j]>Dtype(0.0))
					gradient.mutable_cpu_data()[p_ind*fea_dim + j] = gradient.cpu_data()[p_ind*fea_dim + j]
					- alpha.cpu_data()[i]/static_cast<Dtype>(pairwise_num);
			}
		}
		// negative
		for (int i = 0; i < pairwise_num; ++i){
			n_ind = negative_index.cpu_data()[i];
			for (int j = 0; j < fea_dim; ++j){
				if (p_margin*alpha.cpu_data()[i] - dist_sq_.cpu_data()[i*fea_dim + j]>Dtype(0.0))
					gradient.mutable_cpu_data()[n_ind*fea_dim + j] = gradient.cpu_data()[n_ind*fea_dim + j]
					+ alpha.cpu_data()[i]/static_cast<Dtype>(pairwise_num);
			}
		}
		// set gradient
		caffe_copy(fea_dim*batch_size, gradient.cpu_data(), bout);
	}

	template <typename Dtype>
	void PairWiseSampleLossLayer<Dtype>::Sample_pair(const vector<Blob<Dtype>*> &bottom){
		// sample the pair according the input list
		pairwise_num = 0;
		const Dtype* feature = bottom[0]->cpu_data();
		const Dtype* label = bottom[2]->cpu_data();
		for (int i = 0; i < batch_size && pairwise_num <pairwise_threshold; ++i){
			for (int j = i + 1; j < batch_size && pairwise_num <pairwise_threshold; ++j){
				// success get the pair 
				if (label[i] != label[j]){
					caffe_copy(fea_dim, feature + i*fea_dim, positive_sample.mutable_cpu_data() + pairwise_num*fea_dim);
					caffe_copy(fea_dim, feature + j*fea_dim, negative_sample.mutable_cpu_data() + pairwise_num*fea_dim);
					positive_class.mutable_cpu_data()[pairwise_num] = label[i];
					negative_class.mutable_cpu_data()[pairwise_num] = label[j];
					positive_index.mutable_cpu_data()[pairwise_num] = i;
					negative_index.mutable_cpu_data()[pairwise_num] = j;
					++pairwise_num;
				}
			}
		}
	}

	template <typename Dtype>
	void PairWiseSampleLossLayer<Dtype>::Sample_pair_pseudo(const vector<Blob<Dtype>*> &bottom){
		// get max label first
		Get_max_index(bottom);
		// sample the pair according to the pseudo label
		pairwise_num = 0;
		const Dtype* feature = bottom[0]->cpu_data();
		const Dtype* label = pseudo_label_.cpu_data();
		for (int i = 0; i < batch_size && pairwise_num <pairwise_threshold; ++i){
			for (int j = i + 1; j < batch_size && pairwise_num <pairwise_threshold; ++j){
				// success get the pair 
				if (label[i] != label[j]){
					caffe_copy(fea_dim, feature + i*fea_dim, positive_sample.mutable_cpu_data() + pairwise_num*fea_dim);
					caffe_copy(fea_dim, feature + j*fea_dim, negative_sample.mutable_cpu_data() + pairwise_num*fea_dim);
					positive_class.mutable_cpu_data()[pairwise_num] = label[i];
					negative_class.mutable_cpu_data()[pairwise_num] = label[j];
					positive_index.mutable_cpu_data()[pairwise_num] = i;
					negative_index.mutable_cpu_data()[pairwise_num] = j;
					++pairwise_num;
				}
			}
		}
	}

	template <typename Dtype>
	void PairWiseSampleLossLayer<Dtype>::Get_max_index(const vector<Blob<Dtype>*> &bottom){
		const Dtype* probability = bottom[1]->cpu_data();
		Dtype max_prob;
		Dtype pseudo_label;
		for (int i = 0; i < batch_size; ++i){
			max_prob = probability[i*class_dim];
			pseudo_label = 0;
			for (int j = 1; j < class_dim; j++){
				if (probability[i*class_dim + j]>max_prob){
					max_prob = probability[i*class_dim + j];
					pseudo_label = j;
				}
			}
			pseudo_label_.mutable_cpu_data()[i] = pseudo_label;	
		}
	}

	INSTANTIATE_CLASS(PairWiseSampleLossLayer);
	REGISTER_LAYER_CLASS(PairWiseSampleLoss);

}