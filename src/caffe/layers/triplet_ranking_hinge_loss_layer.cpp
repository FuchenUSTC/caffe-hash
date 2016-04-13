#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

// added by fuchen long for hashing coding
namespace caffe {
	template <typename Dtype>
	void TripletRankingHingeLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		dim = this->layer_param_.triplet_ranking_hinge_loss_param().dim();
		margin = this->layer_param_.triplet_ranking_hinge_loss_param().margin();
		CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
		CHECK_EQ(bottom[1]->channels(), bottom[2]->channels());
		CHECK_EQ(bottom[0]->channels(), dim); //check the dimension
		CHECK_EQ(bottom[1]->channels(), dim);
		CHECK_EQ(bottom[2]->channels(), dim);
		diff_.Reshape(bottom[0]->num(), 1, 1, 1);
		dist_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
		diff_sub_or_si.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1); // F-F+
		diff_sub_or_di.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1); // F-F-
		diff_pow_or_si.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1); // Pow (F-F+)
		diff_pow_or_di.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1); // Pow (F-F-)
		gradient.Reshape(1, bottom[0]->channels(), 1, 1);
	}



	template <typename Dtype>
	void TripletRankingHingeLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top
		){

		const int batchsize = bottom[0]->num();
		int Dimv = batchsize*dim;
		const Dtype* sub_or_si;
		const Dtype* sub_or_di;
		Dtype b = 2;
		Dtype Tripletlosstotal(0.0);

		//The triplet ranking loss
		caffe_sub(Dimv, bottom[0]->cpu_data(), bottom[1]->cpu_data(), diff_sub_or_si.mutable_cpu_data()); // F-F+
		caffe_sub(Dimv, bottom[0]->cpu_data(), bottom[2]->cpu_data(), diff_sub_or_di.mutable_cpu_data()); // F-F-
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
		Tripletlosstotal = Tripletlosstotal / static_cast<Dtype>(bottom[0]->num()); //get the average loss
		top[0]->mutable_cpu_data()[0] = Tripletlosstotal;
	}

	template <typename Dtype>
	void TripletRankingHingeLossLayer<Dtype>::Backward_cpu(
		const vector<Blob<Dtype>*>& top, const vector<bool> &propagate_down,
		const vector<Blob<Dtype>*>& bottom){
		const Dtype* orignalcode;
		const Dtype* similarcode;
		const Dtype* diffrcode;
		if (propagate_down[0]) {
			for (int i = 0; i < 3; ++i) {// for each stream need to get a loss

				int num = bottom[i]->num(); // get the layers' batchsize
				int channels = bottom[i]->channels();// get the layers' channels, channels==codelength
				for (int j = 0; j < num; ++j) // for each batch
				{
					Dtype* bout = bottom[i]->mutable_cpu_diff();// get the 3 bottoms' address, the i th bottom's address
					orignalcode = bottom[0]->cpu_data() + bottom[0]->offset(j);// get  the original image hash approximate code
					similarcode = bottom[1]->cpu_data() + bottom[1]->offset(j);// get the similar image hash approximate code
					diffrcode = bottom[2]->cpu_data() + bottom[2]->offset(j);//get the different image hash approximate code
					if (i == 0)// for the original bottom layer
					{
						if (dist_sq_.cpu_data()[j]>Dtype(0.0))//if the j th batch's loss > 0
						{
							caffe_sub(dim, diffrcode, similarcode, gradient.mutable_cpu_data());// the distance of F- and F+
							caffe_scal(dim, Dtype(2) / Dtype(num), gradient.mutable_cpu_data());// scale the 2/num
						}
						else
							caffe_sub(dim, diffrcode, diffrcode, gradient.mutable_cpu_data());// if the j th batch's loss <=0 ,return 0 vector

					}
					if (i == 1)// for the similar bottom layer
					{
						if (dist_sq_.cpu_data()[j] > Dtype(0.0))// if the j th batch's loss > 0
						{
							caffe_sub(dim, similarcode, orignalcode, gradient.mutable_cpu_data());// the distance of F+ and F
							caffe_scal(dim, Dtype(2) / Dtype(num), gradient.mutable_cpu_data());//scale the 2/num

						}
						else
							caffe_sub(dim, diffrcode, diffrcode, gradient.mutable_cpu_data());// if the j th batch's loss <=0, return 0 vector
					}
					if (i == 2)// for the different bottom layer
					{
						if (dist_sq_.cpu_data()[j] > Dtype(0.0))// if the j th batch's loss > 0
						{
							caffe_sub(dim, orignalcode, diffrcode, gradient.mutable_cpu_data()); // the distance of F and F-
							caffe_scal(dim, Dtype(2) / Dtype(num), gradient.mutable_cpu_data());//scale the 2/num

						}
						else
							caffe_sub(dim, diffrcode, diffrcode, gradient.mutable_cpu_data());// if the j th batch's loss =0 ,return 0 vector
					}
					caffe_scal(dim, Dtype(2.0), gradient.mutable_cpu_data());
					caffe_copy(channels, gradient.cpu_data(), bout + (j*channels));//return the BP vector to the j th batch's bottom
				}
			}

		}
	}


	INSTANTIATE_CLASS(TripletRankingHingeLossLayer);
	REGISTER_LAYER_CLASS(TripletRankingHingeLoss);

}  // namespace caffe