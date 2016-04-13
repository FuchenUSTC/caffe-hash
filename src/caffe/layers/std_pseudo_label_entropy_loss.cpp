#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

// added by Fuchen Long for the weakly supervised learning in 2/29/2016
namespace caffe{
	template <typename Dtype>
	void StdPseudoLabelEntropyLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
		class_num = this->layer_param_.std_pseudo_label_entropy_loss_param().class_num();
		alpha = this->layer_param_.std_pseudo_label_entropy_loss_param().alpha();
		beta = this->layer_param_.std_pseudo_label_entropy_loss_param().beta();
		std_margin = this->layer_param_.std_pseudo_label_entropy_loss_param().std_margin_();
		batch_size = bottom[0]->num();
		//Reshape the variance and the pow_data
		variance_.Reshape(bottom[0]->num(), 1, 1, 1);
		pow_data_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
		exp_sum.Reshape(bottom[0]->num(), 1, 1, 1);
		combine_target.Reshape(batch_size*class_num, 1, 1, 1);
		exp_submax.Reshape(bottom[0]->num(),bottom[0]->channels(), 1, 1);
		max_index.Reshape(bottom[0]->num(), 1, 1, 1);
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		CHECK_EQ(batch_size, bottom[1]->num())
			<< "STD_PSEUDO_LABEL_CROSS_ENTROPY_LOSS layer must have the same label with batch size.";
	}

	template <typename Dtype>
	void StdPseudoLabelEntropyLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
		const Dtype* input_data = bottom[0]->cpu_data();  // The probability
		const Dtype* target = bottom[1]->cpu_data();      // The label
		//get_Std(input_data, batch_size, class_num);       // compute the variance
		//select the low rank to computer the pesudo label
		vector<bool> flag;
		pseudo_num = 0;
		for (int batch_index = 0; batch_index < batch_size; ++batch_index){
			const int labelvalue = static_cast<int>(target[batch_index]);
			int maxindex = 0;
			if (labelvalue != -1){
				if (labelvalue>9){ // The only processed by the noisy trans
					maxindex = labelvalue - 10; //change it to the class_num
					exp_sum.mutable_cpu_data()[pseudo_num] = process_softmax(input_data + batch_index*class_num, class_num,pseudo_num);
					++pseudo_num;
				}
				else{
					maxindex = labelvalue;
				}
				flag.push_back(true); // for the high rank and low variance
			}
			else{
				float MaxNum = input_data[batch_index*class_num]; //high variance but identity through the identity matrix
				for (int k = 1; k < class_num; ++k)
				{
					if (input_data[batch_index*class_num + k]>MaxNum)
					{
						MaxNum = input_data[batch_index*class_num + k];
						maxindex = k;
					}
				}
				flag.push_back(false);
			}
			for (int i = 0; i < class_num; ++i)
			{
				if (i == maxindex)
					combine_target.mutable_cpu_data()[i + batch_index*class_num] = 1;
				else
					combine_target.mutable_cpu_data()[i + batch_index*class_num] = 0;
			}
		}
		//compute the loss
		Dtype loss = 0;
		int pseudo_index = 0;
		const Dtype* data_exp_submax = exp_submax.cpu_data();
		//test
		//std::ofstream test_out("D:\\35-WeeklySupervisedLearning\\CIFAR10\\Network\\ICLR2015\\Network_train_debug\\test_identity\\std_pseudo_prob.txt", ios::out);
		for (int i = 0; i < batch_size; ++i)
		{
			Dtype loss_buff = 0;
			int label = static_cast<int>(target[i]);
			if (label<10){
				for (int j = 0; j < class_num; ++j){
					loss_buff -= (combine_target.cpu_data()[i*class_num + j] == 1) * log(std::max(input_data[i*class_num + j], Dtype(FLT_MIN))) +
						(combine_target.cpu_data()[i*class_num + j] == 0)*log(std::max(1 - input_data[i*class_num + j], Dtype(FLT_MIN)));
					//test_out << input_data[i*class_num + j] << " ";
				}
				//test_out << endl;
				if (flag[i]){
					loss += beta*loss_buff; // For the real label 
				}
				else{
					loss += (1 - alpha - beta)*loss_buff; // For the pseudo label
				}
			}
			else{
				for (int j = 0; j < class_num; ++j){
					loss_buff -= (combine_target.cpu_data()[i*class_num + j] == 1)*log(std::max(data_exp_submax[pseudo_index*class_num + j] / exp_sum.cpu_data()[pseudo_index], Dtype(FLT_MIN))) +
						(combine_target.cpu_data()[i*class_num + j] == 0)*log(std::max(1 - data_exp_submax[pseudo_index*class_num + j] / exp_sum.cpu_data()[pseudo_index], Dtype(FLT_MIN)));
				}
				loss += alpha*loss_buff; // For the trans label
				++pseudo_index;			
			}
		}
		//test_out.close();
		top[0]->mutable_cpu_data()[0] = loss / batch_size;
	}

	template <typename Dtype>
	void StdPseudoLabelEntropyLossLayer<Dtype>::Backward_cpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		if (propagate_down[0]){
			const Dtype* input_data = bottom[0]->cpu_data();
			const Dtype* target = bottom[1]->cpu_data();
			const int count = bottom[0]->count();
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
			const Dtype* data_exp_submax = exp_submax.cpu_data();
			const Dtype* data_exp_sum = exp_sum.cpu_data();
			Dtype gradient;
			int pseudo_index = 0;
			int pseudo_num;
			//compute the gradient
			for (int i = 0; i < batch_size; ++i)
			{
				for (int j = 0; j < class_num; ++j){
					gradient = -(combine_target.cpu_data()[i*class_num + j] == 1) / (std::max(input_data[i*class_num + j], Dtype(FLT_MIN))) +
						(combine_target.cpu_data()[i*class_num + j] == 0) / (std::max(1 - input_data[i*class_num + j], Dtype(FLT_MIN)));
					if (target[i] < 10)
					{
						if (target[i] != -1)
							bottom_diff[i*class_num + j] = beta*gradient; // For the real label
						else
							bottom_diff[i*class_num + j] = (1 - alpha - beta)*gradient; // For the pseudo label

					}
					else{
						gradient = 0;
						pseudo_num = static_cast<int>(std::floor(pseudo_index / 10));
						for (int label_index = 0; label_index < class_num; ++label_index){
							if (label_index == j){
								gradient -= (combine_target.cpu_data()[i*class_num + label_index] == 1) - data_exp_submax[pseudo_num*class_num + j] / data_exp_sum[pseudo_num];
							}
							else{
								gradient -= (combine_target.cpu_data()[i*class_num + label_index] == 0)*(data_exp_submax[pseudo_num*class_num + j] / (data_exp_sum[pseudo_num] - data_exp_submax[pseudo_num*class_num + label_index]))
									- (data_exp_submax[pseudo_num*class_num + j] / data_exp_sum[pseudo_num]);
							} 
						}
						bottom_diff[i*class_num + j] = gradient*alpha;
						++pseudo_index;
					}
				}
			}
			//scale down gradient
			const Dtype loss_weight = top[0]->cpu_diff()[0];
			caffe_scal(count, loss_weight / batch_size, bottom_diff);
		}
	}

	//template <typename Dtype>
	//void StdPseudoLabelEntropyLossLayer<Dtype>::get_Std(const Dtype* bottom_data, 
	//	const int batch_size, const int dim){
	//	const Dtype pow_num = 2.0;
	//	caffe_powx(batch_size*dim, bottom_data, pow_num, pow_data_.mutable_cpu_data());
	//	for (int num = 0; num < batch_size; ++num){
	//		Dtype ave_data = caffe_cpu_asum(dim, bottom_data + num*dim) / dim;
	//		Dtype ave_pow_data = caffe_cpu_asum(dim, pow_data_.cpu_data() + num*dim) / dim;
	//		variance_.mutable_cpu_data()[num] = ave_pow_data - ave_data*ave_data;
	//	}
	//}

	template <typename Dtype>
	Dtype StdPseudoLabelEntropyLossLayer<Dtype>::process_softmax(const Dtype *fVector,
		const int dim,int buff_index){
		Dtype maxnum = fVector[0];
		max_index.mutable_cpu_data()[buff_index] = 0;
		for (int index = 0; index < dim; ++index){
			if (fVector[index]>maxnum){
				maxnum = fVector[index];
				max_index.mutable_cpu_data()[buff_index] = index;
			}
		}
		Dtype* sub_max= new Dtype[dim];
		for (int index = 0; index < dim; ++index){
			sub_max[index] = fVector[index] - maxnum; // change it or not to sub the max
		}
		caffe_exp<Dtype>(dim, sub_max, sub_max);
		caffe_copy(dim, sub_max, exp_submax.mutable_cpu_data() + buff_index*dim);
		Dtype sum = caffe_cpu_asum(dim, sub_max);
		delete[] sub_max;
		return sum;
	}

	template <typename Dtype>
	Dtype StdPseudoLabelEntropyLossLayer<Dtype>::get_submax_exp(const Dtype *fVector,
		const int dim,int index){
		Dtype maxnum = fVector[0];
		for (int index = 0; index < dim; ++index){
			maxnum = std::max(maxnum, fVector[index]);
		}
		return maxnum;
	}

	INSTANTIATE_CLASS(StdPseudoLabelEntropyLossLayer);
	REGISTER_LAYER_CLASS(StdPseudoLabelEntropyLoss);
}