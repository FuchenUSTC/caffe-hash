#include<vector>
#include<iostream>
#include<fstream>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
   template <typename Dtype>
   void LatentGaussianLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*> &bottom,
	   const vector<Blob<Dtype>*> & top){
	   //to get the std_e, ave_e and the hidden number
	   hidden_num = this->layer_param_.latent_guassian_param().hidden_num();
	   std_e = this->layer_param_.latent_guassian_param().std_e();
	   ave_e = this->layer_param_.latent_guassian_param().ave_e();
	   batch_size = bottom[0]->num();
	   //Check the bottom and top number
	   CHECK_EQ(bottom[0]->channels(), bottom[1]->channels()); // the first layer is the ave and the second layer is the standard deviation
	   CHECK_EQ(bottom[0]->count(), bottom[1]->count());
	   CHECK_EQ(bottom[0]->num(), bottom[1]->num());
	   // Reshape the gaussian variable and the top shape
	   gaussian_vector.Reshape(1, hidden_num, 1, 1);
	   
   }

   template <typename Dtype>
   void LatentGaussianLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	   const vector<Blob<Dtype>*>& top){
	   top[0]->Reshape(batch_size, hidden_num, 1, 1);
   }

   template<typename Dtype>
   void LatentGaussianLayer<Dtype>::Forward_cpu(
	   const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	   // Generate the gaussian distribution
	   Dtype* gaussian = gaussian_vector.mutable_cpu_data();
	   const Dtype* gaussian_read = gaussian_vector.cpu_data();
	   caffe_rng_gaussian<Dtype>(hidden_num, ave_e, std_e, gaussian);
	   //Test
	   /*for (int i = 0; i < hidden_num; ++i){
		   std::cout << gaussian_vector.cpu_data()[i] << " ";
	   }
	   std::cout << endl;
	   */
	   const Dtype* ave = bottom[0]->cpu_data();
	   const Dtype* std = bottom[1]->cpu_data();
	   Dtype* hidden_out = top[0]->mutable_cpu_data();
	   // Make the output with the guassian distribution
	   for (int i = 0; i < batch_size; ++i){
		   Dtype* hidden_point = hidden_out + i*hidden_num;
		   const Dtype* std_point = std + i*hidden_num;
		   const Dtype* ave_point = ave + i*hidden_num;
		   for (int j = 0; j < hidden_num; ++j) hidden_point[j] = std_point[j]*gaussian_read[j];
		   caffe_add(hidden_num, hidden_point, ave_point, hidden_point);
	   }
   }

   template<typename Dtype>
   void LatentGaussianLayer<Dtype>::Backward_cpu(
	   const vector<Blob<Dtype>* >& top, const vector<bool>& propagate_down,
	   const vector<Blob<Dtype>* >& bottom){
	   Dtype* gradient_ave = bottom[0]->mutable_cpu_diff();
	   Dtype* gradient_std = bottom[1]->mutable_cpu_diff();
	   caffe_set(bottom[0]->count(), Dtype(1.0), gradient_ave); // the gradient for the average 
	   for (int i = 0; i < batch_size; ++i){
		   caffe_copy(hidden_num, gaussian_vector.mutable_cpu_data(), gradient_std + i*hidden_num); // Set the gradient for the std
	   }
   }

   INSTANTIATE_CLASS(LatentGaussianLayer);
   REGISTER_LAYER_CLASS(LatentGaussian);

}