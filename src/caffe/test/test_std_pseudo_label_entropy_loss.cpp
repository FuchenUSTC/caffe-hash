#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

//this test file tis for the std_pseudo_label_entropy_loss
//Added by Fuchen Long in 3/1/2016

namespace caffe{
template <typename TypeParam>
class StdPseudoLabelEntropyLossLayerTest : public MultiDeviceTest<TypeParam>{
protected:
	StdPseudoLabelEntropyLossLayerTest()
		:blob_bottom_data_(new Blob<Dtype>(10, 10, 1, 1)),
		 blob_bottom_label_(new Blob<Dtype>(10,1,1,1)), 
		 blob_top_loss_(new Blob<Dtype>()){
		//Fill the value in the bottom data
		for (int i = 0; i < 10; ++i){
			for (int j = 0; j < 10; ++j){
				blob_bottom_data_->mutable_cpu_data()[j + i * 10] = -0.883 - 0.001*i + 0.002*j;
				if (j == i)
					blob_bottom_data_->mutable_cpu_data()[j + i * 10] = 1.2311 + i*0.0023- j*0.0013;
			}
		}
		//Fill the label value
		for (int i = 0; i < 10; ++i){
			blob_bottom_label_->mutable_cpu_data()[i] = i + 10;
		}
		//blob_bottom_label_->mutable_cpu_data()[5] = 19;
		blob_bottom_vec_.push_back(blob_bottom_data_);
		blob_bottom_vec_.push_back(blob_bottom_label_);
		blob_top_vec_.push_back(blob_top_loss_);
	}
	
	virtual ~StdPseudoLabelEntropyLossLayerTest(){
		delete blob_bottom_data_;
		delete blob_bottom_label_;
		delete blob_top_loss_;
	}

	void TestForward(){
		LayerParameter layer_param;
		StdPseudoLabelEntropyLossParameter * std_pseudo_label_entropy_loss_param = layer_param.mutable_std_pseudo_label_entropy_loss_param();
		std_pseudo_label_entropy_loss_param->set_alpha(0.7);
		std_pseudo_label_entropy_loss_param->set_beta(0.1);
		std_pseudo_label_entropy_loss_param->set_class_num(10);
		StdPseudoLabelEntropyLossLayer<Dtype> layer_weight_1(layer_param);
		layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		const Dtype loss_weight_1 =
			layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		//Get the loss again with a different object weight
		//check that it is scaled appropriately
		const Dtype KLossWeight = 8.8;
		layer_param.add_loss_weight(KLossWeight);
		StdPseudoLabelEntropyLossLayer<Dtype> layer_weight_2(layer_param);
		layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		const Dtype loss_weight_2 =
			layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		const Dtype kErrorMargin = 1e-5;
		EXPECT_NEAR(loss_weight_1*KLossWeight, loss_weight_2, kErrorMargin);
		//Make sure the loss is non-trivial
		const Dtype kNonTrivialAbsThresh = 1e-1;
		EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
	}

	Blob<Dtype>* const blob_bottom_data_;
	Blob<Dtype>* const blob_bottom_label_;
	Blob<Dtype>* const blob_top_loss_;
	vector<Blob<Dtype>*> blob_bottom_vec_;
	vector<Blob<Dtype>*> blob_top_vec_;

};

TYPED_TEST_CASE(StdPseudoLabelEntropyLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(StdPseudoLabelEntropyLossLayerTest, TestForward){
	this->TestForward();
}

TYPED_TEST(StdPseudoLabelEntropyLossLayerTest, TestGradient){
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	const Dtype kLossWeight = 3.7;
	layer_param.add_loss_weight(kLossWeight);
	StdPseudoLabelEntropyLossParameter * std_pseudo_label_entropy_loss_param = layer_param.mutable_std_pseudo_label_entropy_loss_param();
	std_pseudo_label_entropy_loss_param->set_alpha(0.7);
	std_pseudo_label_entropy_loss_param->set_beta(0.1);
	std_pseudo_label_entropy_loss_param->set_class_num(10);
	StdPseudoLabelEntropyLossLayer<Dtype> layer(layer_param);
	layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
	checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
		this->blob_top_vec_,0);
}

}// namespace caffe