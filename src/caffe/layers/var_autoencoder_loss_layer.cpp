#include <algorithm>
#include <functional>
#include <cfloat>
#include <vector>
#include <iostream>
#include <utility>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

// The varitional autoencoder loss function is added
// by fuchen long in 4/8/2016

namespace caffe{
   template<typename Dtype>
   void VarAutoEncoderLossLayer<Dtype>::LayerSetUp(
   	const vector<Blob<Dtype>*>&bottom, const vector<Blob<Dtype>*>&top){
   	LossLayer<Dtype>::LayerSetUp(bottom, top);
   	CHECK_EQ(bottom[0]->channels(),bottom[1]->channels());
   	CHECK_EQ(bottom[2]->channels(), bottom[3]->channels());
	batch_size = bottom[0]->num();
	count = bottom[0]->count();
   	// Reshape the computer buff
	unit_vector.Reshape(count, 1, 1, 1);
	caffe_set(count, Dtype(1.0), unit_vector.mutable_cpu_data());
   }
   
   template<typename Dtype>
   void VarAutoEncoderLossLayer<Dtype>::Forward_cpu(
	   const vector<Blob<Dtype>*>&bottom, const vector<Blob<Dtype>*> &top){
	   Dtype logpx = 0;
	   Dtype KLD = 0;
	   const Dtype* ave_e = bottom[0]->cpu_data();
	   const Dtype* std_e = bottom[1]->cpu_data();
	   const Dtype* encode_x = bottom[2]->cpu_data();
	   const Dtype* decode_x = bottom[3]->cpu_data();
	   for (int i = 0; i < count; ++i){
		   logpx = logpx - (encode_x[i] * log(decode_x[i]))
			   - (1 - encode_x[i])*log(1 - decode_x[i]);
		   //KLD = KLD + 0.5*(1 + 2 * log(abs(std_e[i])) - ave_e[i] * ave_e[i] -
		   //	   std_e[i] * std_e[i]);
		   KLD = 0;

	   }
	   Dtype loss = (logpx + KLD) / batch_size;
	   top[0]->mutable_cpu_data()[0] = loss;
   }

   template<typename Dtype>
   void VarAutoEncoderLossLayer<Dtype>::Backward_cpu(
	   const vector<Blob<Dtype>*>&top, const vector<bool> &propagate_down,
	   const vector<Blob<Dtype>*>&bottom){

	   Dtype* gradient_ave = bottom[0]->mutable_cpu_diff();
	   Dtype* gradient_std = bottom[1]->mutable_cpu_diff();
	   Dtype* gradient_encode_x = bottom[2]->mutable_cpu_diff();
	   Dtype* gradient_decode_x = bottom[3]->mutable_cpu_diff();
	   // the gradient for average
	   //caffe_copy(count, bottom[0]->mutable_cpu_data(), gradient_ave);
	   //caffe_scal(count, Dtype(-1.0), gradient_ave);
	   // the gradient for std
	   //caffe_div(count, unit_vector.cpu_data(), bottom[1]->cpu_data(), gradient_std);
	   //caffe_sub(count, bottom[1]->cpu_diff(), bottom[1]->cpu_data(), gradient_std);
	   // the gradient for encode x
	   ///for (int i = 0; i < count; ++i){
	   //	   gradient_encode_x[i] = -log(bottom[2]->cpu_data()[i])
	   //		   - log(1 - bottom[2]->cpu_data()[i]);
	   //}
	   // the gradient for decode x
	   for (int i = 0; i < count; ++i){
		   gradient_decode_x[i] = -(bottom[2]->cpu_data()[i] / bottom[3]->cpu_data()[i])
			   + ((1 - bottom[2]->cpu_data()[i]) / (1 - bottom[3]->cpu_data()[i]));
	   }

   }

   INSTANTIATE_CLASS(VarAutoEncoderLossLayer);
   REGISTER_LAYER_CLASS(VarAutoEncoderLoss);
}