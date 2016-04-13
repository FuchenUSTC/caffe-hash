#include "fea_extractor.h"

//added by fuchen long for the hash pipeline 12/2/2015
bool CreateHandle(FeaExtractor<float>* &fea_extractor)
{
	fea_extractor = new FeaExtractor<float>();
	if (fea_extractor == NULL)
	{
		std::cout << "create handle error" << std::endl;
		return false;
	}
	return true;
}

bool ReleaseHandle(FeaExtractor<float>* &fea_extractor)
{
	if (fea_extractor != NULL)
	{
		delete fea_extractor;
		fea_extractor = NULL;
		std::cout << "release handle" << std::endl;
	}
	return true;
}

bool LoadDModel(
	void *fea_extractor,
	//char *mean_path,    // mean.binaryproto
	char *net_path,    // net.txt
	char *model_path, // model
	char *layer_name     // layer_name can be "fc6", "fc7", "fc8", "prob"
	)
{
	FeaExtractor<float> *ptr = (FeaExtractor<float> *)fea_extractor;

	try
	{
		//return ptr->LoadModel(mean_path, net_path, model_path, layer_name);
		return ptr->LoadModel(net_path, model_path, layer_name);
	}
	catch (...)
	{
		std::cout << "LoadDModel error, a serious error occurred" << std::endl;
		return false;
	}
}



bool ExtractFeatureByMergeData(void *fea_extractor, float *merge_fea_data, float *feabuf,int Dim)
{
		FeaExtractor<float> *ptr = (FeaExtractor<float> *)fea_extractor;
		boost::lock_guard<boost::mutex> lock{ *ptr->mutex_ };
		try
		{
			return ptr->ExtractFeatureFromMergeFeature(feabuf, merge_fea_data, Dim);
		}
		catch (...)
		{
			std::cout << "ExtractFeaturesByPath error, a serious error occurred" << std::endl;
			return false;
		}	
}

int GetFeaDim(void *fea_extractor)
{
	FeaExtractor<float> *ptr = (FeaExtractor<float> *)fea_extractor;
	return ptr->GetFeaDim();
}

template <typename Dtype>
FeaExtractor<Dtype>::FeaExtractor()
{
	dim_ = -1;
	mutex_ = NULL;
	dmodel_loader_ = NULL;
	merge_feature_reader_ = NULL;
}

template <typename Dtype>
FeaExtractor<Dtype>::~FeaExtractor()
{
	if (mutex_ != NULL)
		delete mutex_;
	if (dmodel_loader_ != NULL)
		delete dmodel_loader_;
	if (merge_feature_reader_ != NULL)
		delete merge_feature_reader_;

	mutex_ = NULL;
	dmodel_loader_ = NULL;
	merge_feature_reader_ = NULL;

	std::cout << "release fea_extractor" << std::endl;
}

template <typename Dtype>
bool FeaExtractor<Dtype>::LoadModel(
	//string mean_path,    // vgg_mean.binaryproto
	string net_path,    // vgg_net.txt
	string model_path, // vgg.model
	string layer_name     // layer_name can be "fc6", "fc7", "fc8", "prob"
	)
{
	if (mutex_ == NULL)
	{
		mutex_ = new boost::mutex();
	}
	if (dmodel_loader_ == NULL)
	{
		dmodel_loader_ = new DModelLoader<Dtype>();
	}
	if (merge_feature_reader_ == NULL)
	{
		merge_feature_reader_ = new MergeReader<Dtype>();	
	}
	
	if (mutex_ == NULL || dmodel_loader_ == NULL || merge_feature_reader_ == NULL)
	{
		std::cout << "FeaExtractor::LoadModel error, out of memory, init error" << std::endl;
		return false;
	}
	
	if (layer_name.length() > 10)
	{
		std::cout << "FeaExtractor::LoadModel error, invalid layer_name" << std::endl;
		return false;
	}
	strcpy_s(layer_name_, layer_name.c_str());

	// load vgg model
	std::cout << "begin to load model" << std::endl;
	bool succ = dmodel_loader_->LoadModel(net_path, model_path);
	if (!succ) 
	{ 
		std::cout << "FeaExtractor::LoadModel error, failed to load model" << std::endl;
		return false; 
	}
	std::cout << "finish loading model" << std::endl;

	// get the feature dimension
	dim_ = dmodel_loader_->GetFeaDim(layer_name_);
	if (dim_ == -1)
	{
		std::cout << "FeaExtractor::LoadModel error,  failed to get feature dimension" << std::endl;
		return false; 
	}


	return true;
}



template<typename Dtype>
bool FeaExtractor<Dtype>::ExtractFeatureFromMergeFeature(Dtype *feabuf,float *Mergefea,int Dim)
{
	bool succ = merge_feature_reader_->ReadTheMergeFeature(Mergefea, Dim);
	const Dtype *merge_data = merge_feature_reader_->GetMergeData();
	dmodel_loader_->Forward(merge_data);
	bool succed = dmodel_loader_->GetFeatures(feabuf, layer_name_); // copy the features to fea_ buffer
	if (!succed)
	{
		std::cout << "FeaExtractor::ExtractFea error, dmodel_loader_ failed to get features" << std::endl;
		return false;
	}
	
	return true;
}


template class FeaExtractor<float>;
template class FeaExtractor<double>;