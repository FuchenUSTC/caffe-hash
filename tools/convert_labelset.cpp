//Added by fuchen long for creat the label leveldb 2015.11.19
//
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


 
int main(int argc, char **argv) // arg[0] labelList arg[1] outputleveldb arg[2] totalnumber
{
	::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif
	//argv[0] = "D:\\17-MutilabelNusWide\\DataSet\\New_Nus_Sample\\TrainVGGList\\TestVggLabel.txt";
	//argv[1] = "D:\\17-MutilabelNusWide\\DataSet\\New_Nus_Sample\\TrainVGGList\\TestLabelNus_ld";
	//The label Num
	//argv[2] = "159462";
	//argv[2] = "1050";
	//The label dim
	//argv[3] = "21";
	int totalnumber = atoi(argv[3]);
	int labeldim = atoi(argv[4]);
	std::ifstream inlabel(argv[1]);
	//Create new DB
	leveldb::DB* db;
	leveldb::Options options;
	options.create_if_missing = true;
	options.error_if_exists = true;
	leveldb::Status status = leveldb::DB::Open(
		options, argv[2], &db);
	CHECK(status.ok()) << "Failed to open leveldb " << argv[1]
		<< ". Is it already existing?";
	const int kMaxKeyLength = 100;
	char key[kMaxKeyLength];
	std::string value;

	for (int index = 0; index < totalnumber; index++)
	{
		caffe::Datum datum;
		datum.set_channels(labeldim);//For Nus 21 category or triplet
		datum.set_height(1);
		datum.set_width(1);
		if (index == 1000)
			std::cout << "The index 1000 is ";
		for (int j = 0; j < labeldim; j++)
		{
			float label;
			inlabel >> label;
			datum.add_float_data(label);
			if (index == 1000)
				std::cout << label << " ";
		}
		datum.set_label(1);
		datum.SerializeToString(&value);
		_snprintf(key, kMaxKeyLength, "%08d", index);
		db->Put(leveldb::WriteOptions(), std::string(key), value);
		if (index % 10000 == 0)
			std::cout << "triplet:" << index << "\n";

	}

	//delete the db
	delete db;

}