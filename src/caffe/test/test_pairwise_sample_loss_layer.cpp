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


// test the pairwise_sample_loss_layer.
// Added by Fuchen Long 4/27/2016

namespace caffe{
	template <typename TypeParam>
	class PairWiseSampleLossLayerTest : public MultiDeviceTest<TypeParam>{
	protected:
		PairWiseSampleLossLayerTest()
			:blob_bottom_feature_(new Blob<Dtype>(128, 30, 1, 1)),
			blob_bottom_prob_(new Blob<Dtype>(128, 14, 1, 1)),
			blob_bottom_label_(new Blob<Dtype>(128, 1, 1, 1)),
			blob_top_loss_(new Blob<Dtype>()){
			// Filler the value in the feature, prob, layer
			FillerParameter filler_param;
			GaussianFiller<Dtype> filler(filler_param);
			// feature
			filler.Fill(this->blob_bottom_feature_);
			blob_bottom_vec_.push_back(blob_bottom_feature_);
			// prob
			filler.Fill(this->blob_bottom_prob_);
			blob_bottom_vec_.push_back(blob_bottom_prob_);
			// fill the label
			for (int i = 0; i < 128; ++i){
				blob_bottom_label_->mutable_cpu_data()[i] = i % 14;
			}
			blob_bottom_vec_.push_back(blob_bottom_label_);
			// top loss
			blob_top_vec_.push_back(blob_top_loss_);
		}

		virtual ~PairWiseSampleLossLayerTest(){
			delete blob_bottom_feature_;
			delete blob_bottom_prob_;
			delete blob_bottom_label_;
			delete blob_top_loss_;
		}

		void TestForward(){
			// Get the loss without a specific object weight
			LayerParameter layer_param;
			// set some hyper parameter
			PairWiseSampleLossParameter * pairwise_sample_loss_param = layer_param.mutable_pairwise_sample_loss_param();
			pairwise_sample_loss_param->set_margin(1.0);
			pairwise_sample_loss_param->set_num(500);
			pairwise_sample_loss_param->set_noisy_flag(false);
			PairWiseSampleLossLayer<Dtype> layer_weight_1(layer_param);
			layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
			const Dtype loss_weight_1 =
				layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
			// Get the loss again with different object weight
			// check that the scale
			const Dtype kLossWeight = 7.7;
			layer_param.add_loss_weight(kLossWeight);
			PairWiseSampleLossLayer<Dtype> layer_weight_2(layer_param);
			layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
			const Dtype loss_weight_2 =
				layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
			const Dtype kErrorMargin = 1e-5;
			EXPECT_NEAR(loss_weight_1*kLossWeight, loss_weight_2, kErrorMargin);
			// Make sure ther loss is non-trivial 
			const Dtype kNonTrivialAbsThresh = 1e-1;
			EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
		}

		Blob<Dtype>* const blob_bottom_feature_;
		Blob<Dtype>* const blob_bottom_prob_;
		Blob<Dtype>* const blob_bottom_label_;
		Blob<Dtype>* const blob_top_loss_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(PairWiseSampleLossLayerTest, TestDtypesAndDevices);

	TYPED_TEST(PairWiseSampleLossLayerTest, TestForward){
		this->TestForward();
	}

	TYPED_TEST(PairWiseSampleLossLayerTest, TestGradient){
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		const Dtype kLossWeight = 3.7;
		layer_param.add_loss_weight(kLossWeight);
		PairWiseSampleLossParameter * pairwise_sample_loss_param = layer_param.mutable_pairwise_sample_loss_param();
		pairwise_sample_loss_param->set_margin(1.0);
		pairwise_sample_loss_param->set_num(500);
		pairwise_sample_loss_param->set_noisy_flag(false);
		PairWiseSampleLossLayer<Dtype> layer(layer_param);
		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_,0);
	}
}