#ifndef FEA_EXTRACTOR_H_
#define FEA_EXTRACTOR_H_

#include "dmodel_loader.h"
#include "image_reader.h"

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
		string mean_path,    // vgg_mean.binaryproto
		string net_path,    // vgg_net.txt
		string model_path, // vgg.model
		string layer_name     // layer_name can be "fc6", "fc7", "fc8", "prob"
	);

	inline int GetFeaDim() { return dim_; }        // get the dimension of the features

	bool ExtractFea(string image_path, Dtype *feabuf); // extract vgg features
	bool ExtractFea(void *scan0, int width, int height, int stride, int channel, Dtype *feabuf);
	bool ExtractFea(cv::Mat &mat, Dtype *feabuf);
public:
	boost::mutex *mutex_;

private:
	bool ExtractFeaAfterReadImg(Dtype *feabuf);

	DModelLoader<Dtype> *dmodel_loader_; // used for loading vgg model
	ImageReader<Dtype> *image_reader_; // used for reading and cropping images

	char layer_name_[20];      // layer_name can be "fc6", "fc7", "fc8", "prob"
	int dim_;               // the dimension of the vgg_features
};

extern "C"
{
	__declspec(dllexport) bool CreateHandle(FeaExtractor<float>* &fea_extractor);

	__declspec(dllexport) bool ReleaseHandle(FeaExtractor<float>* &fea_extractor);

	__declspec(dllexport) bool LoadDModel(
		void *fea_extractor,
		char *mean_path,    // mean.binaryproto
		char *net_path,    // net.txt
		char *model_path, // model
		char *layer_name     // layer_name can be "fc6", "fc7", "fc8", "prob"
		);

	__declspec(dllexport) bool ExtractFeaturesByPath(void *fea_extractor, char *image_path, float *feabuf);

	__declspec(dllexport) bool ExtractFeaturesByMat(void *fea_extractor, cv::Mat &mat, float *feabuf);

	__declspec(dllexport) bool ExtractFeaturesByData(
		void *fea_extractor, 
		void *scan0, 
		int image_width, 
		int image_height, 
		int image_stride, 
		int channel, 
		float *feabuf);

	__declspec(dllexport) int GetFeaDim(void *fea_extractor);
}


#endif