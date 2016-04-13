#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>
#include <stdint.h>
#include <sys/stat.h>

#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "caffe/proto/caffe.pb.h"

//added by fuchen long for the semi-supervised learning
//and the label is the format int. 
//Note and the dataset is the leveldb.

// port for Win32
#ifdef _MSC_VER
#include <direct.h>
#define snprintf sprintf_s
#endif

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;


uint32_t swap_endian(uint32_t val) {
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	return (val << 16) | (val >> 16);
}

void convert_dataset(const char* image_filename, const char * label_filename,
	const char* db_path){
	//Open the files
	std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
	std::ifstream label_file(label_filename, std::ios::in);
	CHECK(image_file) << "Unable to open file " << image_filename;
	CHECK(label_file) << "Unable to open file " << label_filename;
	//Read the image files and the label
	//file
	uint32_t magic;
	uint32_t num_items;
	uint32_t num_labels;
	uint32_t rows;
	uint32_t cols;

	image_file.read(reinterpret_cast<char*>(&magic), 4);
	magic = swap_endian(magic);
	CHECK_EQ(magic, 2051) << "Incorrect image file magic.";
	image_file.read(reinterpret_cast<char*>(&num_items), 4);
	num_items = swap_endian(num_items);
	image_file.read(reinterpret_cast<char*>(&rows), 4);
	rows = swap_endian(rows);
	image_file.read(reinterpret_cast<char*>(&cols), 4);
	cols = swap_endian(cols);


	// leveldb
	leveldb::DB* db = NULL;
	leveldb::Options options;
	options.error_if_exists = true;
	options.create_if_missing = true;
	options.write_buffer_size = 268435456;
	leveldb::WriteBatch* batch = NULL;

	//Open db
	LOG(INFO) << "Opening leveldb " << db_path;
	leveldb::Status status = leveldb::DB::Open(
		options, db_path, &db);
	CHECK(status.ok()) << "Failed to open leveldb " << db_path
		<< ". Is it already existing?";
	batch = new leveldb::WriteBatch();

	//Storing to leveldb
	int label; //for the format int for the label -1
	int imagename;// for the image name
	char* pixels = new char[rows * cols];
	int count = 0;
	const int kMaxKeyLength = 10;
	char key_cstr[kMaxKeyLength];
	string value;

	Datum datum;
	datum.set_channels(1);
	datum.set_height(rows);
	datum.set_width(cols);
	LOG(INFO) << "A total of " << num_items << " items.";
	LOG(INFO) << "Rows: " << rows << " Cols: " << cols;

	//loop one:
	char** pixels_store = new char *[num_items];
	for (int iter_id = 0; iter_id < num_items; ++iter_id){
		pixels_store[iter_id] = new char[rows * cols];
		image_file.read(pixels_store[iter_id], rows * cols);
	}

	//loop two: 
	for (int item_id = 0; item_id < num_items; ++item_id) {
		label_file >> imagename;
		label_file >> label;
		datum.set_data(pixels_store[imagename], rows*cols);
		datum.set_label(label);
		snprintf(key_cstr, kMaxKeyLength, "%08d", item_id);
		datum.SerializeToString(&value);
		string keystr(key_cstr);
		batch->Put(keystr, value);
		if (++count % 1000 == 0){
			db->Write(leveldb::WriteOptions(), batch);
			delete batch;
			batch = new leveldb::WriteBatch();
		}
	}

	//write the last batch
	if (++count % 1000 == 0){
		db->Write(leveldb::WriteOptions(), batch);
		delete batch;
		delete db;
		LOG(ERROR) << "Processed " << count << " files.";
	}

	delete pixels;
	for (int i = 0; i < num_items; i++)
		delete[] pixels_store[i];
	delete[] pixels_store;
}

int main(int argc, char ** argv){
	if (argc != 4){
		gflags::ShowUsageWithFlagsRestrict(argv[0],
			"examples/mnist/convert_mnist_data");
	}
	else {
		google::InitGoogleLogging(argv[0]);
		convert_dataset(argv[1], argv[2], argv[3]);
	}
}