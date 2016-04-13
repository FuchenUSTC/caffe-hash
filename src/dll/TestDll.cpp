#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include<stdlib.h>
#include<fstream>
#include<ostream>
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "leveldb/db.h"
#include "stdint.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"
#include"merge_feature_reader.h"
#include "fea_extractor.h"
#include "dmodel_loader.h"


int main(int argc, char** argv)
{
	char *Netpath = "D:\\30-VideoHashPipeline\\Model\\Hash\\Hash_256.prototxt.txt";
	char *Modelpath = "D:\\30-VideoHashPipeline\\Model\\Hash\\Hash_256.caffemodel";
	char *Layer_name = "concat1";
	char *Merge_Feature = "D:\\30-VideoHashPipeline\\Model\\Hash\\Check\\Merge_Feature.txt";
	std::ifstream InMergeHash(Merge_Feature);
	FeaExtractor<float> *fea_extract;
	CreateHandle(fea_extract);
	LoadDModel(fea_extract, Netpath, Modelpath, Layer_name);
 	float *merge_pers_fea = new float[8192];
	for (int i = 0; i < 8192; i++)
	{
		InMergeHash>>merge_pers_fea[i];
	}
	float *feabuf = new float[256];
	ExtractFeatureByMergeData(fea_extract, merge_pers_fea, feabuf, 8192);
	for (int i = 0; i < 256; i++)
	{
		std::cout << feabuf[i] << std::endl;
	}
	
}
