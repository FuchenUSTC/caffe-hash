// Added by Fuchen Long for converting the binary feature
// to the leveldb

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

//port for win32
#ifdef _MSC_VER
#define snprintf sprintf_s
#endif

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;
DEFINE_string(backend, "leveldb",
	"The backend {lmdb, leveldb} for storing the result");

int main(int args, char** argv){ // usage: convert_binaryfea_lvdb feature_bin outputdbnum labellist
	::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	// set the path in debug
	argv[1] = "D:\\Clothing1M\\feature_process\\clothing1M_fc8_latent.bin";
	argv[2] = "D:\\Clothing1M\\feature_process\\clothing1M_fc8_latent_lvdb";
	//argv[3] = "D:\\Clothing1M\\feature_process\\ntype_val_label.txt";
	// set the buff size of each lvdb
	leveldb::Options options;
	options.write_buffer_size = 256 * 1024 * 1024; // 256M for each file
	leveldb::DB* db;
	options.create_if_missing = true;
	options.error_if_exists = true;
	leveldb::Status status = leveldb::DB::Open(options, argv[2], &db);
	CHECK(status.ok()) << "Failed to open leveldb" << argv[1] << ". Is it already existing?";
	const int kMaxKeyLength = 100;
	char key[kMaxKeyLength];
	std::string value;
	// read the file of the binary files 
	std::ifstream feature_in(argv[1], ios::binary); // to set the number of the feature
	std::ifstream label_in(argv[3], ios::in); // to set the label
	int feature_num = 0;
	int feature_dim = 0;
	int type_size = 0;
	int label;
	feature_in.read((char*)(&feature_num), sizeof(feature_num));
	feature_in.read((char*)(&feature_dim), sizeof(feature_dim));
	feature_in.read((char*)(&type_size), sizeof(type_size));
	// And to read the next feature for the training
	for (int i = 0; i < feature_num; ++i){
		caffe::Datum datum;
		datum.set_channels(feature_dim);
		datum.set_height(1);
		datum.set_width(1);
		Blob<float> vector;
		vector.Reshape(1, feature_dim, 1, 1);
		feature_in.read((char*)(vector.mutable_cpu_data()), type_size*feature_dim);
		for (int j = 0; j < feature_dim; ++j) datum.add_float_data(vector.cpu_data()[j]);
		//label_in >> label;
		datum.set_label(0);
		datum.SerializePartialToString(&value);
		_snprintf(key, kMaxKeyLength, "%08d", i);
		db->Put(leveldb::WriteOptions(), std::string(key), value);
		if (i % 10000 == 0) std::cout << "Have converted " << i << " binary feature.\n";
	}

	// delete the db
	delete db;
	feature_in.close();
	label_in.close();
}