#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

//added by fuchen long for the unpooling layer test 1/19/2016
namespace caffe{

template <typename TypeParam>
class UnPoolingLayerTest : public MultiDeviceTest<TypeParam>{
	typedef typename TypeParam::Dtype Dtype;

protected:
	UnPoolingLayerTest()
		:blob_bottom_(new Blob<Dtype>()),
		blob_top_(new Blob<Dtype>()){}
	virtual void SetUp(){
		Caffe::set_random_seed(1701);
		blob_bottom_->Reshape(2, 3, 10, 12);
		//fill the value
		FillerParameter filler_param;
		GaussianFiller<Dtype> filler(filler_param);
		filler.Fill(this->blob_bottom_);
		blob_bottom_vec_.push_back(blob_bottom_);
		blob_top_vec_.push_back(blob_top_);
	}

	virtual ~UnPoolingLayerTest(){
		delete blob_bottom_;
		delete blob_top_;
	}
	Blob<Dtype>* const blob_top_;
	Blob<Dtype>* const blob_bottom_;
	vector<Blob<Dtype>*> blob_bottom_vec_;
	vector<Blob<Dtype>*> blob_top_vec_;

	//Test for 2*2 square pooling layer
	void TestForwardSquare(){
		LayerParameter layer_param;
		//share the hyper parameter in the pooling layer
		PoolingParameter* unpooling_param = layer_param.mutable_pooling_param();
		unpooling_param->set_kernel_size(2);
		unpooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
		//In this case, we needn't to distinguish the max pooling or the average pooling, and the
		//unpooling must loss some information.
		const int num = 2;
		const int channels = 2;
		blob_bottom_->Reshape(num, channels, 2, 3);
		// Input: 2x 2 channels of:
		//     [1 2 5]
		//     [9 4 1]
		//This is for the test the unpooling layer
		for (int i = 0; i < 6 * num * channels; i += 6) {
			blob_bottom_->mutable_cpu_data()[i + 0] = 1;
			blob_bottom_->mutable_cpu_data()[i + 1] = 2;
			blob_bottom_->mutable_cpu_data()[i + 2] = 5;
			blob_bottom_->mutable_cpu_data()[i + 3] = 9;
			blob_bottom_->mutable_cpu_data()[i + 4] = 4;
			blob_bottom_->mutable_cpu_data()[i + 5] = 1;
		}
		UnPoolingLayer<Dtype> layer(layer_param);
		layer.SetUp(blob_bottom_vec_, blob_top_vec_);
		EXPECT_EQ(blob_top_->num(), num);
		EXPECT_EQ(blob_top_->channels(), channels);
		EXPECT_EQ(blob_top_->height(), 3);
		EXPECT_EQ(blob_top_->width(), 4);
		layer.Forward(blob_bottom_vec_, blob_top_vec_);
		//The Expected Output 2*2 channels of 
		// [1 1 2 5]
		// [1 1 1 1]
		// [9 4 1 1]
		for (int i = 0; i < 12 * num*channels; i += 12){
			EXPECT_EQ(blob_top_->cpu_data()[i + 0], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 1], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 2], 2);
			EXPECT_EQ(blob_top_->cpu_data()[i + 3], 5);
			EXPECT_EQ(blob_top_->cpu_data()[i + 4], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 5], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 6], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 7], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 8], 9);
			EXPECT_EQ(blob_top_->cpu_data()[i + 9], 4);
			EXPECT_EQ(blob_top_->cpu_data()[i + 10], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 11], 1);
		}
	}

	//Test for 3*2 rectangular pooling layer with kerner_h> kernel_w
	void TestForwardRectHigh(){
		LayerParameter layer_param;
		PoolingParameter *unpooling_param = layer_param.mutable_pooling_param();
		unpooling_param->set_kernel_h(3);
		unpooling_param->set_kernel_w(2);
		unpooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
		const int num = 2;
		const int channels = 2;
		blob_bottom_->Reshape(num, channels, 2, 3);
		// Input: 2x 2 channels of:
		//     [1 2 5]
		//     [2 4 1]
		//This is for the test the unpooling layer
		for (int i = 0; i < 6 * num * channels; i += 6) {
			blob_bottom_->mutable_cpu_data()[i + 0] = 1;
			blob_bottom_->mutable_cpu_data()[i + 1] = 2;
			blob_bottom_->mutable_cpu_data()[i + 2] = 5;
			blob_bottom_->mutable_cpu_data()[i + 3] = 2;
			blob_bottom_->mutable_cpu_data()[i + 4] = 4;
			blob_bottom_->mutable_cpu_data()[i + 5] = 1;
		}
		UnPoolingLayer<Dtype> layer(layer_param);
		layer.SetUp(blob_bottom_vec_, blob_top_vec_);
		EXPECT_EQ(blob_top_->num(), num);
		EXPECT_EQ(blob_top_->channels(), channels);
		EXPECT_EQ(blob_top_->height(), 4);
		EXPECT_EQ(blob_top_->width(), 4);
		layer.Forward(blob_bottom_vec_, blob_top_vec_);
		//Expected output 2*2 channels of:
		//[1 1 2 5]
		//[1 1 1 1]
		//[1 1 1 1]
		//[2 2 1 1] // this is not the to the original matrix
		for (int i = 0; i < 16 * num*channels; i += 16)
		{
			EXPECT_EQ(blob_top_->cpu_data()[i + 0], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 1], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 2], 2);
			EXPECT_EQ(blob_top_->cpu_data()[i + 3], 5);
			EXPECT_EQ(blob_top_->cpu_data()[i + 4], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 5], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 6], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 7], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 8], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 9], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 10], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 11], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 12], 2);
			EXPECT_EQ(blob_top_->cpu_data()[i + 13], 2);
			EXPECT_EQ(blob_top_->cpu_data()[i + 14], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 15], 1);
		}
	}

	//Test fot the kernel_w > kernel_h
	void TestForwardRectWide()
	{
		LayerParameter layer_param;
		PoolingParameter *unpooling_param = layer_param.mutable_pooling_param();
		unpooling_param->set_kernel_h(2);
		unpooling_param->set_kernel_w(3);
		unpooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
		const int num = 2;
		const int channels = 2;
		blob_bottom_->Reshape(num, channels, 2, 3);
		// Input: 2x 2 channels of:
		//     [1 2 5]
		//     [2 4 1]
		//This is for the test the unpooling layer
		for (int i = 0; i < 6 * num * channels; i += 6) {
			blob_bottom_->mutable_cpu_data()[i + 0] = 1;
			blob_bottom_->mutable_cpu_data()[i + 1] = 2;
			blob_bottom_->mutable_cpu_data()[i + 2] = 5;
			blob_bottom_->mutable_cpu_data()[i + 3] = 2;
			blob_bottom_->mutable_cpu_data()[i + 4] = 4;
			blob_bottom_->mutable_cpu_data()[i + 5] = 1;
		}
		UnPoolingLayer<Dtype> layer(layer_param);
		layer.SetUp(blob_bottom_vec_, blob_top_vec_);
		EXPECT_EQ(blob_top_->num(), num);
		EXPECT_EQ(blob_top_->channels(), channels);
		EXPECT_EQ(blob_top_->height(), 3);
		EXPECT_EQ(blob_top_->width(), 5);
		layer.Forward(blob_bottom_vec_, blob_top_vec_);
		//Expected output: 2*2 channels of:
		// [1 1 1 2 5]
		// [1 1 1 1 1]
		// [2 2 1 1 1]
		for (int i = 0; i < 15 * num*channels; i += 15)
		{
			EXPECT_EQ(blob_top_->cpu_data()[i + 0], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 1], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 2], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 3], 2);
			EXPECT_EQ(blob_top_->cpu_data()[i + 4], 5);
			EXPECT_EQ(blob_top_->cpu_data()[i + 5], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 6], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 7], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 8], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 9], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 10], 2);
			EXPECT_EQ(blob_top_->cpu_data()[i + 11], 2);
			EXPECT_EQ(blob_top_->cpu_data()[i + 12], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 13], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 14], 1);
		}
	}

	//Test for the stride = kernel_h
	void TestForwardSquareEqule(){
		LayerParameter layer_param;
		//share the hyper parameter in the pooling layer
		PoolingParameter* unpooling_param = layer_param.mutable_pooling_param();
		unpooling_param->set_kernel_size(2);
		unpooling_param->set_stride(2);
		unpooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
		//In this case, we needn't to distinguish the max pooling or the average pooling, and the
		//unpooling must loss some information.
		const int num = 2;
		const int channels = 2;
		blob_bottom_->Reshape(num, channels, 2, 2);
		// Input: 2x 2 channels of:
		//     [1 2]
		//     [9 4]
		//This is for the test the unpooling layer
		for (int i = 0; i < 4 * num * channels; i += 4) {
			blob_bottom_->mutable_cpu_data()[i + 0] = 1;
			blob_bottom_->mutable_cpu_data()[i + 1] = 2;
			blob_bottom_->mutable_cpu_data()[i + 2] = 9;
			blob_bottom_->mutable_cpu_data()[i + 3] = 4;
		}
		UnPoolingLayer<Dtype> layer(layer_param);
		layer.SetUp(blob_bottom_vec_, blob_top_vec_);
		EXPECT_EQ(blob_top_->num(), num);
		EXPECT_EQ(blob_top_->channels(), channels);
		EXPECT_EQ(blob_top_->height(), 4);
		EXPECT_EQ(blob_top_->width(), 4);
		layer.Forward(blob_bottom_vec_, blob_top_vec_);
		//The Expected Output 2*2 channels of 
		// [1 1 2 2]
		// [1 1 2 2]
		// [9 9 4 4]
		// [9 9 4 4]
		for (int i = 0; i < 16 * num*channels; i += 16){
			EXPECT_EQ(blob_top_->cpu_data()[i + 0], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 1], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 2], 2);
			EXPECT_EQ(blob_top_->cpu_data()[i + 3], 2);
			EXPECT_EQ(blob_top_->cpu_data()[i + 4], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 5], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 6], 2);
			EXPECT_EQ(blob_top_->cpu_data()[i + 7], 2);
			EXPECT_EQ(blob_top_->cpu_data()[i + 8], 9);
			EXPECT_EQ(blob_top_->cpu_data()[i + 9], 9);
			EXPECT_EQ(blob_top_->cpu_data()[i + 10], 4);
			EXPECT_EQ(blob_top_->cpu_data()[i + 11], 4);
			EXPECT_EQ(blob_top_->cpu_data()[i + 12], 9);
			EXPECT_EQ(blob_top_->cpu_data()[i + 13], 9);
			EXPECT_EQ(blob_top_->cpu_data()[i + 14], 4);
			EXPECT_EQ(blob_top_->cpu_data()[i + 15], 4);
		}
	}

	//Test for the average unpooling
	void TestForwardSquareAve(){
		LayerParameter layer_param;
		//share the hyper parameter in the pooling layer
		PoolingParameter* unpooling_param = layer_param.mutable_pooling_param();
		unpooling_param->set_kernel_size(2);
		unpooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
		//In this case, we needn't to distinguish the max pooling or the average pooling, and the
		//unpooling must loss some information.
		const int num = 2;
		const int channels = 2;
		blob_bottom_->Reshape(num, channels, 2, 3);
		// Input: 2x 2 channels of:
		//     [1 2 5]
		//     [9 4 1]
		//This is for the test the unpooling layer
		for (int i = 0; i < 6 * num * channels; i += 6) {
			blob_bottom_->mutable_cpu_data()[i + 0] = 1;
			blob_bottom_->mutable_cpu_data()[i + 1] = 2;
			blob_bottom_->mutable_cpu_data()[i + 2] = 5;
			blob_bottom_->mutable_cpu_data()[i + 3] = 9;
			blob_bottom_->mutable_cpu_data()[i + 4] = 4;
			blob_bottom_->mutable_cpu_data()[i + 5] = 1;
		}
		UnPoolingLayer<Dtype> layer(layer_param);
		layer.SetUp(blob_bottom_vec_, blob_top_vec_);
		EXPECT_EQ(blob_top_->num(), num);
		EXPECT_EQ(blob_top_->channels(), channels);
		EXPECT_EQ(blob_top_->height(), 3);
		EXPECT_EQ(blob_top_->width(), 4);
		layer.Forward(blob_bottom_vec_, blob_top_vec_);
		//The Expected Output 2*2 channels of 
		// [1 3/2 7/2 5]
		// [5  4   3  3]
		// [9 13/2 5/2 1]
		Dtype epsilon = 1e-5;
		for (int i = 0; i < 12 * num*channels; i += 12){
			EXPECT_NEAR(blob_top_->cpu_data()[i + 0], 1.0, epsilon);
			EXPECT_NEAR(blob_top_->cpu_data()[i + 1], 3.0 / 2, epsilon);
			EXPECT_NEAR(blob_top_->cpu_data()[i + 2], 7.0/2, epsilon);
			EXPECT_NEAR(blob_top_->cpu_data()[i + 3], 5.0, epsilon);
			EXPECT_NEAR(blob_top_->cpu_data()[i + 4], 5.0, epsilon);
			EXPECT_NEAR(blob_top_->cpu_data()[i + 5], 4.0, epsilon);
			EXPECT_NEAR(blob_top_->cpu_data()[i + 6], 3.0, epsilon);
			EXPECT_NEAR(blob_top_->cpu_data()[i + 7], 3.0, epsilon);
			EXPECT_NEAR(blob_top_->cpu_data()[i + 8], 9.0, epsilon);
			EXPECT_NEAR(blob_top_->cpu_data()[i + 9], 13.0/2, epsilon);
			EXPECT_NEAR(blob_top_->cpu_data()[i + 10], 5.0/2, epsilon);
			EXPECT_NEAR(blob_top_->cpu_data()[i + 11], 1.0, epsilon);
		}
	}
};

TYPED_TEST_CASE(UnPoolingLayerTest, TestDtypesAndDevices);

TYPED_TEST(UnPoolingLayerTest, TestForwardMax){
	this->TestForwardSquare();
	this->TestForwardRectHigh();
	this->TestForwardRectWide();
	this->TestForwardSquareEqule();
}

TYPED_TEST(UnPoolingLayerTest, TestGradientMax){
	typedef typename TypeParam::Dtype Dtype;
	for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
		for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
			LayerParameter layer_param;
			PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
			pooling_param->set_kernel_h(kernel_h);
			pooling_param->set_kernel_w(kernel_w);
			pooling_param->set_stride(2);
			pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
			PoolingLayer<Dtype> layer(layer_param);
			GradientChecker<Dtype> checker(1e-4, 1e-2);
			checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
				this->blob_top_vec_);
		}
	}
}

TYPED_TEST(UnPoolingLayerTest, TestForwardAve){
	this->TestForwardSquareAve();
}

TYPED_TEST(UnPoolingLayerTest, TestGradientAve){
	typedef typename TypeParam::Dtype Dtype;
	for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
		for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
			LayerParameter layer_param;
			PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
			pooling_param->set_kernel_h(kernel_h);
			pooling_param->set_kernel_w(kernel_w);
			pooling_param->set_stride(2);
			pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
			PoolingLayer<Dtype> layer(layer_param);
			GradientChecker<Dtype> checker(1e-4, 1e-2);
			checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
				this->blob_top_vec_);
		}
	}
}

}
