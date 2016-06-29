#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

// Added by Fuchen Long in 6/21/2016
// For the video hashing learning

namespace caffe {
	template <typename Dtype>
	void TripletClipHingeLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		dim = this->layer_param_.triplet_clip_hinge_loss_param().dim();
		margin = this->layer_param_.triplet_clip_hinge_loss_param().margin();
		frame_num = this->layer_param_.triplet_clip_hinge_loss_param().frame_num();
		lamda = this->layer_param_.triplet_clip_hinge_loss_param().lamda();
		int ave_num = bottom[0]->num() / frame_num;
		batch = bottom[0]->num() / frame_num;
		CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
		CHECK_EQ(bottom[1]->channels(), bottom[2]->channels());
		CHECK_EQ(bottom[0]->channels(), dim); //check the dimension
		CHECK_EQ(bottom[1]->channels(), dim);
		CHECK_EQ(bottom[2]->channels(), dim);
		CHECK_EQ((bottom[0]->num())% frame_num, 0) <<
			"TRIPLET_CLIP_HINGE_LOSS_LAYER: the batchsize must div frame_num.";
		diff_.Reshape(ave_num, 1, 1, 1);
		dist_sq_.Reshape(ave_num, 1, 1, 1);
		diff_sub_or_si.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1); // F-F+
		diff_sub_or_di.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1); // F-F-
		diff_pow_or_si.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1); // Pow (F-F+)
		diff_pow_or_di.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1); // Pow (F-F-)
		ave_or.Reshape(ave_num, bottom[0]->channels(), 1, 1);// Ave(F)
		ave_si.Reshape(ave_num, bottom[0]->channels(), 1, 1);// Ave(F+)
		ave_di.Reshape(ave_num, bottom[0]->channels(), 1, 1);// Ave(F-)
		sub_or.Reshape(ave_num*(frame_num - 1), bottom[0]->channels(), 1, 1);// SUB(F)
		sub_si.Reshape(ave_num*(frame_num - 1), bottom[0]->channels(), 1, 1);// SUB(F+)
		sub_di.Reshape(ave_num*(frame_num - 1), bottom[0]->channels(), 1, 1);// SUB(F-)
		pow_sub_or.Reshape(ave_num*(frame_num - 1), bottom[0]->channels(), 1, 1);// SUB(F)
		pow_sub_si.Reshape(ave_num*(frame_num - 1), bottom[0]->channels(), 1, 1);// SUB(F+)
		pow_sub_di.Reshape(ave_num*(frame_num - 1), bottom[0]->channels(), 1, 1);// SUB(F-)
		gradient_triplet.Reshape(1, bottom[0]->channels(), 1, 1);
		gradient_structure.Reshape(1, bottom[0]->channels(), 1, 1);
		gradient.Reshape(1, bottom[0]->channels(), 1, 1);
	}

	template<typename Dtype>
	void TripletClipHingeLossLayer<Dtype>::
		average_hashing(const vector<Blob<Dtype>*>& bottom){

		int batch_size = bottom[0]->num() / frame_num;
		caffe_set(batch_size*dim, Dtype(0.0), ave_or.mutable_cpu_data());
		caffe_set(batch_size*dim, Dtype(0.0), ave_si.mutable_cpu_data());
		caffe_set(batch_size*dim, Dtype(0.0), ave_di.mutable_cpu_data());

		for (int i = 0; i < batch_size; ++i){
			for (int j = 0; j < frame_num; ++j){
				int index = i*frame_num*dim + j*dim;
				caffe_add(dim, bottom[0]->cpu_data() + index,
					ave_or.cpu_data() + i*dim, ave_or.mutable_cpu_data() + i*dim);
				caffe_add(dim, bottom[1]->cpu_data() + index,
					ave_si.cpu_data() + i*dim, ave_si.mutable_cpu_data() + i*dim);
				caffe_add(dim, bottom[2]->cpu_data() + index,
					ave_di.cpu_data() + i*dim, ave_di.mutable_cpu_data() + i*dim);
			}
			caffe_scal(dim, 1 / Dtype(frame_num), ave_or.mutable_cpu_data() + i*dim);
			caffe_scal(dim, 1 / Dtype(frame_num), ave_si.mutable_cpu_data() + i*dim);
			caffe_scal(dim, 1 / Dtype(frame_num), ave_di.mutable_cpu_data() + i*dim);
		}
	}

	template<typename Dtype>
	Dtype TripletClipHingeLossLayer<Dtype>::compute_tripletloss(int batchsize,
		int Dimv){
		Dtype Tripletlosstotal(0.0);
		const Dtype* sub_or_si;
		const Dtype* sub_or_di;
		//The triplet ranking loss
		caffe_sub(Dimv, ave_or.cpu_data(), ave_si.cpu_data(), diff_sub_or_si.mutable_cpu_data()); // F-F+
		caffe_sub(Dimv, ave_or.cpu_data(), ave_di.cpu_data(), diff_sub_or_di.mutable_cpu_data()); // F-F-
		caffe_powx(Dimv, diff_sub_or_si.cpu_data(), Dtype(2.0), diff_pow_or_si.mutable_cpu_data());		  //Pow
		caffe_powx(Dimv, diff_sub_or_di.cpu_data(), Dtype(2.0), diff_pow_or_di.mutable_cpu_data());       //Pow
		for (int n = 0; n < batchsize; n++)
		{
			sub_or_si = diff_pow_or_si.cpu_data() + diff_pow_or_si.offset(n);
			sub_or_di = diff_pow_or_di.cpu_data() + diff_pow_or_di.offset(n);
			Dtype result1 = 0;
			Dtype result2 = 0;
			result1 = caffe_cpu_asum(dim, sub_or_si);
			result2 = caffe_cpu_asum(dim, sub_or_di);
			Dtype loss(0.0);
			loss = std::max(margin + result1 - result2, Dtype(0));// compute the loss
			diff_.mutable_cpu_data()[n] = loss; // save the loss[i]
		}
		for (int k = 0; k < batchsize; k++)
		{

			dist_sq_.mutable_cpu_data()[k] = diff_.cpu_data()[k];// save the loss[i] for BP
			Tripletlosstotal += dist_sq_.cpu_data()[k];
		}
		return Tripletlosstotal / static_cast<Dtype>(batchsize);
	}

	template<typename Dtype>
	Dtype TripletClipHingeLossLayer<Dtype>::
		compute_structureloss(const vector<Blob<Dtype>*>& bottom){

			Dtype Structureloss(0.0);
			int batch_size = bottom[0]->num() / frame_num;
			for (int i = 0; i < batch_size; ++i){
				for (int j = 0; j < frame_num - 1; ++j){
					int index_1 = i*frame_num*dim + j*dim;
					int index_2 = i*frame_num*dim + (j + 1)*dim;
					int direct = i*(frame_num - 1)*dim + j*dim;
					caffe_sub(dim, bottom[0]->cpu_data() + index_1,
						bottom[0]->cpu_data() + index_2, sub_or.mutable_cpu_data() + direct);
					caffe_sub(dim, bottom[1]->cpu_data() + index_1,
						bottom[1]->cpu_data() + index_2, sub_si.mutable_cpu_data() + direct);
					caffe_sub(dim, bottom[2]->cpu_data() + index_1,
						bottom[2]->cpu_data() + index_2, sub_di.mutable_cpu_data() + direct);
					// pow 
					caffe_powx(dim, sub_or.cpu_data() + direct, Dtype(2.0), pow_sub_or.mutable_cpu_data() + direct);
					caffe_powx(dim, sub_si.cpu_data() + direct, Dtype(2.0), pow_sub_si.mutable_cpu_data() + direct);
					caffe_powx(dim, sub_di.cpu_data() + direct, Dtype(2.0), pow_sub_di.mutable_cpu_data() + direct);
					// plus
					Structureloss += (caffe_cpu_asum(dim, pow_sub_or.cpu_data() + direct)
						+ caffe_cpu_asum(dim, pow_sub_si.cpu_data() + direct)
						+ caffe_cpu_asum(dim, pow_sub_di.cpu_data() + direct));
				}
			}
			return Structureloss / (batch_size*(frame_num - 1) * 3);
		}

	template <typename Dtype>
	void TripletClipHingeLossLayer<Dtype>::
		compute_gradient_structure(int index,int hash_pos){
			int inner_pos = hash_pos % frame_num;
			int sub_pos = (hash_pos / frame_num)*(frame_num - 1);
			if (index == 0){
				if (inner_pos == 0) 
					caffe_copy(dim, sub_or.cpu_data() + sub_pos*dim, 
					gradient_structure.mutable_cpu_data());
				else if (inner_pos == frame_num - 1){
					caffe_copy(dim, sub_or.cpu_data() + (sub_pos + inner_pos - 1)*dim,
						gradient_structure.mutable_cpu_data());
					caffe_scal(dim, Dtype(-1), gradient_structure.mutable_cpu_data());
				}
				else{
					caffe_sub(dim, sub_or.cpu_data() + (sub_pos + inner_pos)*dim,
						sub_or.cpu_data() + (sub_pos + inner_pos - 1)*dim,
						gradient_structure.mutable_cpu_data());
				}
				caffe_scal(dim, Dtype(2.0) / (Dtype(3)*batch*(frame_num - 1)),
					gradient_structure.mutable_cpu_data());
			}
			if (index == 1){
				if (inner_pos == 0)
					caffe_copy(dim, sub_si.cpu_data() + sub_pos*dim,
					gradient_structure.mutable_cpu_data());
				else if (inner_pos == frame_num - 1){
					caffe_copy(dim, sub_si.cpu_data() + (sub_pos + inner_pos - 1)*dim,
						gradient_structure.mutable_cpu_data());
					caffe_scal(dim, Dtype(-1), gradient_structure.mutable_cpu_data());
				}
				else{
					caffe_sub(dim, sub_si.cpu_data() + (sub_pos + inner_pos)*dim,
						sub_si.cpu_data() + (sub_pos + inner_pos - 1)*dim, 
						gradient_structure.mutable_cpu_data());
				}
				caffe_scal(dim, Dtype(2.0) / (Dtype(3)*batch*(frame_num - 1)),
					gradient_structure.mutable_cpu_data());
			}
			if (index == 2){
				if (inner_pos == 0)
					caffe_copy(dim, sub_di.cpu_data() + sub_pos*dim,
					gradient_structure.mutable_cpu_data());
				else if (inner_pos == frame_num - 1){
					caffe_copy(dim, sub_di.cpu_data() + (sub_pos + inner_pos - 1)*dim,
						gradient_structure.mutable_cpu_data());
					caffe_scal(dim, Dtype(-1), gradient_structure.mutable_cpu_data());
				}
				else{
					caffe_sub(dim, sub_di.cpu_data() + (sub_pos + inner_pos)*dim,
						sub_di.cpu_data() + (sub_pos + inner_pos - 1)*dim, 
						gradient_structure.mutable_cpu_data());
				}
				caffe_scal(dim, Dtype(2.0) / (Dtype(3)*batch*(frame_num - 1)),
					gradient_structure.mutable_cpu_data());
			}
		}

	template <typename Dtype>
	void TripletClipHingeLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top
		){

		const int batchsize = bottom[0]->num() / frame_num;
		int Dimv = batchsize*dim;
		const Dtype* sub_or_si;
		const Dtype* sub_or_di;
		Dtype b = 2;
		Dtype Tripletlosstotal(0.0);
		Dtype Structureloss(0.0);
		Dtype Totalloss(0.0);

		// Average the frame hashing to clip hashing
		average_hashing(bottom);

		// The triplet loss 
		Tripletlosstotal = compute_tripletloss(batchsize, Dimv);

		// The Structure loss
		Structureloss = compute_structureloss(bottom);

		// The Total loss
		Totalloss = lamda * Tripletlosstotal + (1 - lamda)*Structureloss;
		top[0]->mutable_cpu_data()[0] = Totalloss;
	}


	template <typename Dtype>
	void TripletClipHingeLossLayer<Dtype>::Backward_cpu(
		const vector<Blob<Dtype>*>& top, const vector<bool> &propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		const Dtype* orignalcode;
		const Dtype* similarcode;
		const Dtype* diffrcode;
		if (propagate_down[0]) {
			for (int i = 0; i < 3; ++i) {// for each stream need to get a loss
				int num = bottom[i]->num();
				int channels = bottom[i]->channels();
				for (int j = 0; j < num; ++j){
					Dtype* bout = bottom[i]->mutable_cpu_diff();// get the 3 bottoms' address, the i th bottom's address
					orignalcode = ave_or.cpu_data() + (j / num)*dim;
					similarcode = ave_si.cpu_data() + (j / num)*dim;
					diffrcode = ave_di.cpu_data() + (j / num)*dim;
					if (i == 0){
						if (dist_sq_.cpu_data()[j/frame_num]>Dtype(0.0)){
							caffe_sub(dim, diffrcode, similarcode, 
								gradient_triplet.mutable_cpu_data());// the distance of F- and F+
							caffe_scal(dim, Dtype(2) / Dtype(num), 
								gradient_triplet.mutable_cpu_data());
						}
						else
							caffe_sub(dim, diffrcode, diffrcode, 
							gradient_triplet.mutable_cpu_data());
						compute_gradient_structure(i, j);
						caffe_scal(dim, lamda, gradient_triplet.mutable_cpu_data());
						caffe_scal(dim, Dtype(1.0)-lamda, gradient_structure.mutable_cpu_data());
						caffe_add(dim, gradient_triplet.cpu_data(), 
							gradient_structure.cpu_data(), gradient.mutable_cpu_data());
					}
					if (i == 1){
						if (dist_sq_.cpu_data()[j/frame_num] > Dtype(0.0)){
							caffe_sub(dim, similarcode, orignalcode, 
								gradient_triplet.mutable_cpu_data());// the distance of F+ and F
							caffe_scal(dim, Dtype(2) / Dtype(num), 
								gradient_triplet.mutable_cpu_data());
						}
						else
							caffe_sub(dim, diffrcode, diffrcode, 
							gradient_triplet.mutable_cpu_data());
						compute_gradient_structure(i, j);
						caffe_scal(dim, lamda, gradient_triplet.mutable_cpu_data());
						caffe_scal(dim, Dtype(1.0) - lamda, gradient_structure.mutable_cpu_data());
						caffe_add(dim, gradient_triplet.cpu_data(),
							gradient_structure.cpu_data(), gradient.mutable_cpu_data());
					}
					if (i == 2){
						if (dist_sq_.cpu_data()[j/frame_num] > Dtype(0.0)){
							caffe_sub(dim, orignalcode, diffrcode, 
								gradient_triplet.mutable_cpu_data()); 
							caffe_scal(dim, Dtype(2) / Dtype(num), 
								gradient_triplet.mutable_cpu_data());
						}
						else
							caffe_sub(dim, diffrcode, diffrcode, 
							gradient_triplet.mutable_cpu_data());
						compute_gradient_structure(i, j);
						caffe_scal(dim, lamda, gradient_triplet.mutable_cpu_data());
						caffe_scal(dim, Dtype(1.0) - lamda, gradient_structure.mutable_cpu_data());
						caffe_add(dim, gradient_triplet.cpu_data(),
							gradient_structure.cpu_data(), gradient.mutable_cpu_data());
					}
					caffe_scal(dim, Dtype(2.0), gradient.mutable_cpu_data());
					caffe_copy(channels, gradient.cpu_data(), bout + (j*channels));//return the BP vector to the j th batch's bottom
				}
			}
		}
	}


	INSTANTIATE_CLASS(TripletClipHingeLossLayer);
	REGISTER_LAYER_CLASS(TripletClipHingeLoss);

}  // namespace caffe