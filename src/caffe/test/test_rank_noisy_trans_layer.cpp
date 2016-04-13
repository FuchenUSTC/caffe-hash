#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe{

template <typename TypeParam>
class RankNoisyTransLayerTest : public MultiDeviceTest<TypeParam>{
	typedef typename TypeParam::Dtype Dtype;
	protected:
		RankNoisyTransLayerTest()
			:blob_bottom_data_(new Blob<Dtype>(20, 10, 1, 1)),
			 blob_bottom_label_(new Blob<Dtype>(20,1,1,1)),
			 blob_top_data_(new Blob<Dtype>()),
		     blob_top_label_(new Blob<Dtype>()){
			//Fill the values
			FillerParameter filler_param;
			GaussianFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_bottom_data_);
			blob_bottom_vec_.push_back(blob_bottom_data_);
			//Fill the label
			for (int i = 0; i < 20; ++i){
				blob_bottom_label_->mutable_cpu_data()[i] = i % 10;
			}
			blob_bottom_vec_.push_back(blob_bottom_label_);
			blob_top_vec_.push_back(blob_top_data_);
			blob_top_vec_.push_back(blob_top_label_);
		}
		virtual ~RankNoisyTransLayerTest(){ 
			delete blob_bottom_data_;
			delete blob_bottom_label_;
			delete blob_top_data_;
		}
		Blob<Dtype>* const blob_bottom_data_;
		Blob<Dtype>* const blob_top_data_;
		Blob<Dtype>* const blob_bottom_label_;
		Blob<Dtype>* const blob_top_label_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(RankNoisyTransLayerTest, TestDtypesAndDevices);

TYPED_TEST(RankNoisyTransLayerTest, TestSetUp){
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	RankNoisyTransParameter* rank_noisy_trans_param =
		layer_param.mutable_rank_noisy_trans_param();
	rank_noisy_trans_param->set_num_output(10);
	rank_noisy_trans_param->set_var_margin_(0.4);
	rank_noisy_trans_param->mutable_weight_filler()->set_type("identity");
	shared_ptr<RankNoisyTransLayer<Dtype> > layer(
		new RankNoisyTransLayer<Dtype>(layer_param));
	layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	EXPECT_EQ(this->blob_top_data_->num(), 20);
	EXPECT_EQ(this->blob_top_data_->height(), 1);
	EXPECT_EQ(this->blob_top_data_->width(), 1);
	EXPECT_EQ(this->blob_top_data_->channels(), 10);
}

TYPED_TEST(RankNoisyTransLayerTest, TestForward) {
	typedef typename TypeParam::Dtype Dtype;
	bool IS_VALID_CUDA = false;
	if (Caffe::mode() == Caffe::CPU ||
		sizeof(Dtype) == 4 || IS_VALID_CUDA) {
		LayerParameter layer_param;
		RankNoisyTransParameter* rank_noisy_trans_param =
			layer_param.mutable_rank_noisy_trans_param();
		rank_noisy_trans_param->set_num_output(10);
		rank_noisy_trans_param->set_var_margin_(0.4);
		rank_noisy_trans_param->mutable_weight_filler()->set_type("identity");
		shared_ptr<RankNoisyTransLayer<Dtype> > layer(
			new RankNoisyTransLayer<Dtype>(layer_param));
		layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		const Dtype* data = this->blob_top_data_->cpu_data();
		const Dtype* input_data = this->blob_bottom_data_->cpu_data();
		const int count = this->blob_top_data_->count();
		for (int i = 0; i < count; ++i) {
			EXPECT_GE(data[i], input_data[i]);
		}
	}
	else {
		LOG(ERROR) << "Skipping test due to old architecture.";
	}
}

TYPED_TEST(RankNoisyTransLayerTest, TestGradient) {
	typedef typename TypeParam::Dtype Dtype;
	bool IS_VALID_CUDA = false;
	if (Caffe::mode() == Caffe::CPU ||
		sizeof(Dtype) == 4 || IS_VALID_CUDA) {
		LayerParameter layer_param;
		RankNoisyTransParameter* rank_noisy_trans_param =
			layer_param.mutable_rank_noisy_trans_param();
		rank_noisy_trans_param->set_num_output(10);
		rank_noisy_trans_param->set_var_margin_(0.4);
		rank_noisy_trans_param->mutable_weight_filler()->set_type("identity");
		RankNoisyTransLayer<Dtype> layer(layer_param);
		GradientChecker<Dtype> checker(1e-3, 1e-3);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_,0);
	}
	else {
		LOG(ERROR) << "Skipping test due to old architecture.";
	}
}

}