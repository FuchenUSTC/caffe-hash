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

// Added by Fcuhen Long in 6/21/2016
// To test the triplet clip layer 
// for video hashing learning

namespace caffe{

	template <typename TypeParam>
	class TripletClipHingeLossLayerTest : public MultiDeviceTest<TypeParam>{
	protected:
		TripletClipHingeLossLayerTest()
			:blob_bottom_data_original_(new Blob<Dtype>(14, 48, 1, 1)),
			blob_bottom_data_similar_(new Blob<Dtype>(14, 48, 1, 1)),
			blob_bottom_data_different_(new Blob<Dtype>(14, 48, 1, 1)),
			blob_top_loss_(new Blob<Dtype>()){
			//File the value in the original, similar and different layer
			FillerParameter filler_param;
			GaussianFiller<Dtype> filler(filler_param);
			//original 
			filler.Fill(this->blob_bottom_data_original_);
			blob_bottom_vec_.push_back(blob_bottom_data_original_);
			//similar
			filler.Fill(this->blob_bottom_data_similar_);
			blob_bottom_vec_.push_back(blob_bottom_data_similar_);
			//different
			filler.Fill(this->blob_bottom_data_different_);
			blob_bottom_vec_.push_back(blob_bottom_data_different_);
			//top loss
			blob_top_vec_.push_back(blob_top_loss_);
		}

		virtual ~TripletClipHingeLossLayerTest(){
			delete blob_bottom_data_original_;
			delete blob_bottom_data_similar_;
			delete blob_bottom_data_different_;
			delete blob_top_loss_;
		}


		void TestForward(){
			//Get the loss without a specific object weight
			LayerParameter layer_param;
			//set some hyper parameter
			TripletClipHingeLossParameter * triplet_clip_hinge_loss_param 
				= layer_param.mutable_triplet_clip_hinge_loss_param();
			triplet_clip_hinge_loss_param->set_dim(48);
			triplet_clip_hinge_loss_param->set_frame_num(7);
			triplet_clip_hinge_loss_param->set_margin(0.6);
			triplet_clip_hinge_loss_param->set_lamda(0.0);
			TripletClipHingeLossLayer<Dtype> layer_weight_1(layer_param);
			layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
			const Dtype loss_weight_1 =
				layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

			//Get the loss again with a different object weight;
			//check that it is scaled appropriately
			const Dtype kLossWeight = 7.7;
			layer_param.add_loss_weight(kLossWeight);
			TripletClipHingeLossLayer<Dtype> layer_weight_2(layer_param);
			layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
			const Dtype loss_weight_2 =
				layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
			const Dtype kErrorMargin = 1e-5;
			EXPECT_NEAR(loss_weight_1*kLossWeight, loss_weight_2, kErrorMargin);
			//Make sure the loss is non-trivial
			const Dtype kNonTrivialAbsThresh = 1e-1;
			EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
		}


		Blob<Dtype>* const blob_bottom_data_original_;
		Blob<Dtype>* const blob_bottom_data_similar_;
		Blob<Dtype>* const blob_bottom_data_different_;
		Blob<Dtype>* const blob_top_loss_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(TripletClipHingeLossLayerTest, TestDtypesAndDevices);

	TYPED_TEST(TripletClipHingeLossLayerTest, TestForward){
		this->TestForward();
	}

	TYPED_TEST(TripletClipHingeLossLayerTest, TestGradient){
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		const Dtype kLossWeight = 3.7;
		layer_param.add_loss_weight(kLossWeight);
		TripletClipHingeLossParameter * triplet_clip_hinge_loss_param
			= layer_param.mutable_triplet_clip_hinge_loss_param();
		triplet_clip_hinge_loss_param->set_dim(48);
		triplet_clip_hinge_loss_param->set_frame_num(7);
		triplet_clip_hinge_loss_param->set_margin(0.6);
		triplet_clip_hinge_loss_param->set_lamda(0.0);
		TripletClipHingeLossLayer<Dtype> layer(layer_param);
		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_);
	}


}// namespace caffe
