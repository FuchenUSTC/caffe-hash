#ifndef FEA_EXTRACTOR_H_
#define FEA_EXTRACTOR_H_

#include "dmodel_loader.h"
#include "merge_feature_reader.h"

#include <boost/thread.hpp>
#include <boost/chrono.hpp>
using boost::shared_ptr;

template <typename Dtype>
class __declspec(dllexport) FeaExtractor
{
public:
	FeaExtractor();
	~FeaExtractor();
	bool LoadModel(
		//string mean_path,    // vgg_mean.binaryproto
		string net_path,    // hash_net.txt
		string model_path, // hash.model
		string layer_name     // layer_name can be "fc6", "fc7", "fc8", "prob" or for hash the concate
	);

	inline int GetFeaDim() { return dim_; }        // get the dimension of the features


	bool ExtractFeatureFromMergeFeature(Dtype *feabuf, float *Mergefea, int Dim);
public:
	boost::mutex *mutex_;

private:
	//bool ExtractFeaAfterReadImg(Dtype *feabuf);
	DModelLoader<Dtype> *dmodel_loader_; // used for loading hash model
	MergeReader<Dtype> *merge_feature_reader_; // used for reading reading the feature

	char layer_name_[20];      // layer_name can be "fc6", "fc7", "fc8", "prob"
	int dim_;               // the dimension of the approximate hash code 
};

extern "C"
{
	__declspec(dllexport) bool CreateHandle(FeaExtractor<float>* &fea_extractor);

	__declspec(dllexport) bool ReleaseHandle(FeaExtractor<float>* &fea_extractor);

	__declspec(dllexport) bool LoadDModel(
		void *fea_extractor,
		//char *mean_path,    // mean.binaryproto
		char *net_path,    // net.txt
		char *model_path, // model
		char *layer_name     // layer_name can be "fc6", "fc7", "fc8", "prob"
		);
	__declspec(dllexport) bool ExtractFeatureByMergeData(void *fea_extractor, float *merge_fea_data, float *feabuf,int Dim);


	__declspec(dllexport) int GetFeaDim(void *fea_extractor);
}


#endif