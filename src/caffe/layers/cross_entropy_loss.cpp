#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

//added by Fuchen Long for the weekly supervised learning
namespace caffe{
	template <typename Dtype>
	void CrossEntropyLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector <Blob<Dtype>*>& top
		)
	{
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		class_num = this->layer_param_.cross_entropy_loss_param().class_num();
		label_buff.Reshape(bottom[1]->num(), class_num, 1, 1);
		CHECK_EQ(bottom[0]->count(), bottom[1]->num()*class_num) <<
			"CROSS_ENTROPY_LOSS layer inputs must have the same count.";
	}

	template<typename Dtype>
	void CrossEntropyLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top
		)
	{
		const int count = bottom[0]->count();
		const int num = bottom[0]->num();
		const Dtype* input_data = bottom[0]->cpu_data();
		const Dtype* label = bottom[1]->cpu_data();
		//make the target label vector
		for (int i = 0; i < num; ++i){
			int single_label = static_cast<int>(label[i]);
			for (int j = 0; j < class_num; ++j) {
				if (j == single_label) label_buff.mutable_cpu_data()[i*class_num + j] = 1;
				else label_buff.mutable_cpu_data()[i*class_num + j] = 0;
			}
		}
		const Dtype* target = label_buff.cpu_data();
		//compute the cross entropy loss
		Dtype loss = 0;
		for (int i = 0; i < count; i++)
		{
			loss -= (target[i]==1) * log(input_data[i]) + (target[i]==0)*log(1 - input_data[i]);
		}
		top[0]->mutable_cpu_data()[0] = loss / num;
	}

	template<typename Dtype>
	void CrossEntropyLossLayer<Dtype>::Backward_cpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom)
	{
		if (propagate_down[1]) {
			LOG(FATAL) << this->type()
				<< " Layer cannot backpropagate to label inputs.";
		}
		if (propagate_down[0]){
			//compute the back gradietn
			const int count = bottom[0]->count();
			const int num = bottom[0]->num();
			const Dtype* target = label_buff.cpu_data();
			const Dtype* input_data = bottom[0]->cpu_data();
		    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
			for (int i = 0; i < count; i++)
			{
				bottom_diff[i] = -(target[i] == 1) / input_data[i] + (target[i] == 0) / (1 - input_data[i]);
			}
			//scale down gradient
			const Dtype loss_weight = top[0]->cpu_diff()[0];
			caffe_scal(count, loss_weight / num, bottom_diff);
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(CrossEntropyLossLayer);
#endif

	INSTANTIATE_CLASS(CrossEntropyLossLayer);
	REGISTER_LAYER_CLASS(CrossEntropyLoss);
}
