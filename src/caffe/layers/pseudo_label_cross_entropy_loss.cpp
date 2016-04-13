#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

// added by Fuchen Long for the weekly supervised learning in 1/12/2016
namespace caffe{
	template <typename Dtype>
	void PseudoLabelCrossEntropyLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top
		)
	{
		low_iter = this->layer_param_.pseudo_label_cross_entropy_loss_param().low_iter();
		high_iter = this->layer_param_.pseudo_label_cross_entropy_loss_param().high_iter();
		class_num = this->layer_param_.pseudo_label_cross_entropy_loss_param().class_num();
        alpha = this->layer_param_.pseudo_label_cross_entropy_loss_param().alpha();
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		batch_size = bottom[0]->num();
		CHECK_EQ(batch_size, bottom[1]->num())
			<< "PSEUDO_LABEL_CROSS_ENTROPY_LOSS layer must have the same label with batch size.";
		combine_target.Reshape(batch_size*class_num, 1, 1, 1);
		if (blobs_.size() > 0)
		{
			LOG(INFO) << "Skipping parameter initialization";
		}
		else
		{
			this->blobs_.resize(1);
			vector<int> number(1);
			number[0] = 1;
			this->blobs_[0].reset(new Blob<Dtype>(number));
			Dtype *iter = blobs_[0]->mutable_cpu_data();
			iter[0] = 0;
		}

	}

	template <typename Dtype>
	void PseudoLabelCrossEntropyLossLayer<Dtype>::BigProb_select(
		const int label, int index, const vector<Blob<Dtype>*>& bottom)
	{
		class_num = this->layer_param_.pseudo_label_cross_entropy_loss_param().class_num();
		const Dtype* input_data = bottom[0]->cpu_data();
		Dtype maxnum = input_data[0];
		Dtype sum = maxnum;
		vector<bool> flag;
		int maxindex = 0;
		if (label != -1) // the label is the real label make the vector
		{
			maxindex = label;
			flag.push_back(true);
		}
		else // choose the bigest to the label vector
		{
			for (int k = 1; k < class_num; k++)
			{
				if (input_data[k + index*class_num]>maxnum)
				{
					maxnum = input_data[k + index*class_num];
					maxindex = k;
				}
				sum = sum + input_data[k + index*class_num];
			}
			flag.push_back(false);
		}
		if (!(sum == sum))
		{
			LOG(INFO) << "After the select pseudo label, come some diverge!";
			std::cout << sum << endl;
			std::system("pause");
		}
		for (int i = 0; i < class_num; i++)
		{
			if (i == maxindex)
				combine_target.mutable_cpu_data()[i + index*class_num] = 1;
			else
				combine_target.mutable_cpu_data()[i + index*class_num] = 0;
		}

	}

	template <typename Dtype>
	Dtype PseudoLabelCrossEntropyLossLayer<Dtype>::Lamda_update()
	{
		Dtype Lamda;
		//Dtype* iteration = this->blobs_[0].mutable_cpu_data();
		Dtype iterationtime = this->blobs_[0]->mutable_cpu_data()[0];
		//Dtype iterationtime = 1;
		if (iterationtime < low_iter)
		{
			if (iterationtime == 0)
				LOG(INFO) << "The Pseudo Loss parameter alpha will set to 0 from " << iterationtime
				<< " iteration to " << low_iter << " interations";
			Lamda = 0;
		}
		else if (iterationtime < high_iter)
		{
			if (iterationtime == low_iter)
				LOG(INFO) << "The Pseudo Loss parameter alpha will slowly change from " << iterationtime
				<< " iterations" << high_iter << " interations";
			Lamda = alpha*(iterationtime - low_iter) / (high_iter - low_iter);
		}
		else
		{
			if (iterationtime == high_iter)
				LOG(INFO) << "The Pseudo Loss parameter alpha will set to " << alpha << " from " << iterationtime
				<< " iterations to the end interations";
			Lamda = alpha;
		}
		return Lamda;
	}

	template <typename Dtype>
	void PseudoLabelCrossEntropyLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top
		)
	{
		const int classNum = this->layer_param_.pseudo_label_cross_entropy_loss_param().class_num();
		Dtype lamda = Lamda_update();
		const int count = bottom[0]->count();
		const int num = bottom[0]->num();
		const Dtype* input_data = bottom[0]->cpu_data(); //The probability
		const Dtype* target = bottom[1]->cpu_data();     //The label
		vector<bool> flag;
		//Select the pseudo label or check the real label
		for (int batch_index = 0; batch_index < num; batch_index++)
		{
			const int labelvalue = static_cast<int>(target[batch_index]);
			int maxindex = 0;
			//BigProb_select(labelvalue, batch_index, bottom);
			if (labelvalue != -1)
			{
				maxindex = labelvalue;
				flag.push_back(true);
			}
			else
			{
				float MaxNum = input_data[batch_index*classNum];
				for (int k = 1; k < classNum; k++)
				{
					if (input_data[batch_index*classNum + k]>MaxNum)
					{
						MaxNum = input_data[batch_index*classNum + k];
						maxindex = k;
					}
				}
				flag.push_back(false);
			}
			for (int i = 0; i < classNum; i++)
			{
				if (i == maxindex)
					combine_target.mutable_cpu_data()[i + batch_index*classNum] = 1;
				else
					combine_target.mutable_cpu_data()[i + batch_index*classNum] = 0;
			}
		}
			//compute the cross entropy loss, for no lamda the condition.
			//Dtype loss = 0;
			//for (int i = 0; i < count; i++)
			//{
			//	loss -= (combine_target.cpu_data()[i] == 1) * log(std::max(input_data[i], Dtype(FLT_MIN))) + (combine_target.cpu_data()[i] == 0)*log(std::max(1 - input_data[i], Dtype(FLT_MIN)));
			//}
			//
			//top[0]->mutable_cpu_data()[0] = loss / num;
		
		//if you set the lamda and to use this compute
		Dtype loss = 0;
		for (int i = 0; i < num; i++)
		{
			Dtype loss_buff = 0;
			for (int j = 0; j < classNum; j++)
			{
				loss_buff -= (combine_target.cpu_data()[i*classNum + j] == 1) * log(std::max(input_data[i*classNum + j], Dtype(FLT_MIN))) + 
					(combine_target.cpu_data()[i*classNum + j] == 0)*log(std::max(1 - input_data[i*classNum + j], Dtype(FLT_MIN)));
			}
			if (flag[i])
				loss += (1-lamda)*loss_buff;
			else
				loss += lamda*loss_buff;
		}
		top[0]->mutable_cpu_data()[0] = loss / num;
	}

	template<typename Dtype>
	void PseudoLabelCrossEntropyLossLayer<Dtype>::Backward_cpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom)
	{
		Dtype lamda = Lamda_update();
		const int  classNum = this->layer_param_.pseudo_label_cross_entropy_loss_param().class_num();
		if (propagate_down[1]) {
			LOG(FATAL) << this->type()
				<< " Layer cannot backpropagate to label inputs.";
		}
		if (propagate_down[0]){
			//compute the back gradietn
			const int count = bottom[0]->count();
			const int num = bottom[0]->num();
			const Dtype* input_data = bottom[0]->cpu_data();
			const Dtype* target = bottom[1]->cpu_data();
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
			Dtype gradient;
			//compute the gradient, and this is for no lamda the condition
			//for (int i = 0; i < count; i++)
			//{
			//	bottom_diff[i] = -(combine_target.cpu_data()[i] == 1) / (std::max(input_data[i], Dtype(FLT_MIN))) + (combine_target.cpu_data()[i] == 0) / (std::max(1 - input_data[i], Dtype(FLT_MIN)));
			//}

			//if for the lamda and to use this compute the gradient
			for (int i = 0; i < num; i++)
			{
				for (int j = 0; j < classNum; j++)
				{
					gradient = -(combine_target.cpu_data()[i*classNum + j] == 1) / (std::max(input_data[i*classNum + j], Dtype(FLT_MIN))) + 
						(combine_target.cpu_data()[i*classNum + j] == 0) / (std::max(1 - input_data[i*classNum + j], Dtype(FLT_MIN)));
					if (target[i] != -1)
						bottom_diff[i*classNum + j] = (1 - lamda)*gradient;
					else
						bottom_diff[i*classNum + j] = lamda*gradient;
				}
			}
			//scale down gradient
			const Dtype loss_weight = top[0]->cpu_diff()[0];
			caffe_scal(count, loss_weight / num, bottom_diff);
		}
		//change the iter
		//iter++;
		Dtype* iterationtime = this->blobs_[0]->mutable_cpu_data();
		iterationtime[0]++;
	} 

#ifdef CPU_ONLY
	STUB_GPU(PseudoLabelCrossEntropyLossLayer);
#endif


	INSTANTIATE_CLASS(PseudoLabelCrossEntropyLossLayer);
	REGISTER_LAYER_CLASS(PseudoLabelCrossEntropyLoss);
}