#include "dmodel_loader.h"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

// added by fuchen long for the hash pipeline in 12/2/2015
template <typename Dtype>
DModelLoader<Dtype>::DModelLoader()  //Default the imput layer name is the "data"
{
	data_layer_name_ = "data";
	
}

template <typename Dtype>
DModelLoader<Dtype>::~DModelLoader()
{
	std::cout << "model release" << std::endl;
}

template <typename Dtype>
bool DModelLoader<Dtype>::LoadModel(string hash_net_path, string hash_model_path)
{
	//::google::InitGoogleLogging("extract vgg");
	Caffe::set_mode(Caffe::CPU);

	feature_extraction_net_ = boost::shared_ptr<Net<Dtype> >(new Net<Dtype>(hash_net_path, caffe::TEST));
	feature_extraction_net_->CopyTrainedLayersFrom(hash_model_path);

	//caffe::BlobProto blob_proto;
	//caffe::ReadProtoFromBinaryFileOrDie(vgg_mean_path.c_str(), &blob_proto);
	//data_mean_.FromProto(blob_proto);

	if (!feature_extraction_net_->has_blob(data_layer_name_))
	{
		std::cout << "DModelLoader::LoadModel error, no data layer" << std::endl;
		return false;
	}

	data_layer_ = feature_extraction_net_->blob_by_name(data_layer_name_);

	batch_size_ = data_layer_->num();
	channels_ = data_layer_->channels();
	width_ = data_layer_->width();
	height_ = data_layer_->height();
	count_ = batch_size_ * channels_ * width_ * height_;

	//mean_width_ = data_mean_.width();
	//mean_height_ = data_mean_.height();

	data_sub_mean_.Reshape(batch_size_, channels_, height_, width_);

	if (batch_size_ != 1)
	{
		std::cout << "DModelLoader::LoadModel error, batch_size should be equal to 1 (input_dim: 1)" << std::endl;
		return false;
	}

	return true;
}




template <typename Dtype>
void DModelLoader<Dtype>::Forward(const Dtype *Merge_data)
{
	std::vector<Blob<Dtype>*> input_vec;
	//SubMean(image_data, data_mean_.cpu_data(), data_sub_mean_.mutable_cpu_data());
	caffe::caffe_copy(count_, Merge_data, data_sub_mean_.mutable_cpu_data());
	memcpy(data_layer_->mutable_cpu_data(), data_sub_mean_.cpu_data(), sizeof(Dtype)* count_);
	feature_extraction_net_->Forward(input_vec);
}

template <typename Dtype>
bool DModelLoader<Dtype>::GetFeatures(Dtype *fea, const char *layer_name)
{
	if (!feature_extraction_net_->has_blob(layer_name))
	{
		std::cout << "DModelLoader::GetFeatures error, Unknown layer_name " << layer_name << " in the network "<< std::endl;
		return false;
	}
	if (fea == NULL)
	{
		std::cout << "DModelLoader::GetFeatures error, fea == NULL" << endl;
		return false;
	}

	const boost::shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net_
		->blob_by_name(layer_name);

	int batch_size = feature_blob->num(); // batch_size must be 1
	int dim_features = feature_blob->count() / batch_size;

	const Dtype *feature_blob_data = feature_blob->cpu_data() + feature_blob->offset(0);
	for (int i = 0; i < dim_features; i++)
	{
		fea[i] = feature_blob_data[i];
	}

	return true;
}

template <typename Dtype>
bool DModelLoader<Dtype>::GetFeatures(Dtype *fea, string layer_name)
{
	return GetFeatures(fea, layer_name.c_str());
}

template <typename Dtype>
int DModelLoader<Dtype>::GetFeaDim(string layer_name)
{
	if (!feature_extraction_net_->has_blob(layer_name))
	{
		std::cout << "DModelLoader::GetFeaDim error, Unknown layer_name " << layer_name << " in the network "<< std::endl;
		return -1;
	}

	const boost::shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net_
		->blob_by_name(layer_name);

	int batch_size = feature_blob->num();
	int dim_features = feature_blob->count() / batch_size;

	return dim_features;
}

template class DModelLoader<float>;
template class DModelLoader<double>;