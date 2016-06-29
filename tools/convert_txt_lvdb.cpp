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
	//argv[1] = "D:\\45-VideoHashing\\CCV\\Code\\matlab\\ccv_test_fc6.txt";
	//argv[2] = "D:\\45-VideoHashing\\CCV\\Code\\matlab\\ccv_test_fea_fc6_lvdb";
	//argv[3] = "500";
	//argv[4] = "4096";
	//argv[3] = "D:\\Clothing1M\\fc6_feature\\label\\noisy_train_label.txt";

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
	std::ifstream feature_in(argv[1], ios::in); // to set the number of the feature
	std::vector<int> label;

	int feature_num = atoi(argv[3]);
	int feature_dim = atoi(argv[4]);


	// set the feature label
	if (args == 6){
		std::ifstream label_in(argv[5], ios::in); // to set the label
		for (int i = 0; i < feature_num; ++i){
			int index;
			label_in >> index;
			label.push_back(index);
		}
		label_in.close();
	}
	else{
		for (int i = 0; i < feature_num; ++i) label.push_back(0);
	}

	// And to read the next feature for the training
	for (int i = 0; i < feature_num; ++i){
		caffe::Datum datum;
		datum.set_channels(feature_dim);
		datum.set_height(1);
		datum.set_width(1);
		float feature;
		for (int j = 0; j < feature_dim; ++j){
			feature_in >> feature;
			datum.add_float_data(feature);
		}
		datum.set_label(label[i]);
		datum.SerializePartialToString(&value);
		_snprintf(key, kMaxKeyLength, "%08d", i);
		db->Put(leveldb::WriteOptions(), std::string(key), value);
		if (i % 10000 == 0) std::cout << "Have converted " << i << " txt feature.\n";
	}

	// delete the db
	delete db;
	feature_in.close();
}