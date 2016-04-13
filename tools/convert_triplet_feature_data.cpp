//added by fuchen long 2015-8-11
#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "leveldb/db.h"
#include "stdint.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"


//void read_feature(int index, float original[di][], int dim, caffe::Datum& datum) {
//	
//  std::ifstream feature1(feature_list1[index], std::ios::in);
//  std::ifstream feature2(feature_list2[index], std::ios::in);
//  std::ifstream feature3(feature_list3[index], std::ios::in);
//  float value;
//  for(int i=0;i<dim;i++)
//  {
//    feature1>>value;
//	datum.add_float_data(value);
//  }
//
//  for(int i=dim;i<2*dim;i++)
//  {
//    feature2>>value;
//	datum.add_float_data(value);
//  }
//  
//  for(int i=2*dim;i<3*dim;i++)
//  {
//	  feature3>>value;
//	  datum.add_float_data(value);
//  }
//  feature1.close();
//  feature2.close();
//  feature3.close();
//}

void convert_dataset(const char* feature_list_filename1, const char*feature_list_filename2, const char*feature_list_filename3, 
        const char* db_filename, int feature_dim,int num) {
  // Open files
  std::ifstream feature_list_in1(feature_list_filename1, std::ios::in); // original 
  std::ifstream feature_list_in2(feature_list_filename2, std::ios::in);// similar
  std::ifstream feature_list_in3(feature_list_filename3, std::ios::in);// differernt
  //std::ifstream pair_list_in(pair_list_filename, std::ios::in);
  CHECK(feature_list_in1) << "Unable to open file " << feature_list_filename1; // added by fuchen long
  CHECK(feature_list_in2) << "Unable to open file " << feature_list_filename2;
  CHECK(feature_list_in3) << "Unable to open file " << feature_list_filename3;
 // CHECK(pair_list_in) << "Unable to open file " << pair_list_filename;
  
  //read metadata
  //std::vector<std::string> feature_list1;
  //std::vector<std::string> feature_list2;
  //std::vector<std::string> feature_list3;
  const int  dim = feature_dim;
  const int number = num;
  float feature;
  /*float original[number][dim];
  float similar[number][dim];
  float different[number][dim];
  float feature;
  int count = 0;
  

  for (int i = 0; i < number; i++)
  {
	  for (int k = 0; k < dim; k++)
	  {
		  feature_list_in1 >> original[i][k];
	  }
	  if (i % 10000 == 0)
		  std::cout << "original:have load " << i << "features" << "\n";
  }
	   
  for (int i = 0; i < number; i++)
  {
	  for (int k = 0; k < dim; k++)
	  {
		  feature_list_in2 >> similar[i][k];
	  }
	  if (i % 10000 == 0)
		  std::cout << "similar:have load " << i << "features" << "\n";
  }

  for (int i = 0; i < number; i++)
  {
	  for (int k = 0; k < dim; k++)
	  {
		  feature_list_in3 >> different[i][k];
	  }
	  if (i % 10000 == 0)
		  std::cout << "different:have load " << i << "features" << "\n";
  }*/
  //
  //while (feature_list_in2 >> feature)
  //{
	 // if (feature.size() != 0)
		//  feature_list2.push_back(feature);
  //}
  //while (feature_list_in3 >> feature)
  //{
	 // if (feature.size() != 0)
		//  feature_list3.push_back(feature);
  //}

  //int index1;
  //int index2;

  //std::vector<std::pair<int, int>> pair_list;
  //while(pair_list_in>>index1>>index2)
  //{
  //  pair_list.push_back(std::make_pair(index1, index2));
  //  //LOG(INFO)<<"INDEX:"<<index1<<'\t'<<index2;
  //}

 /* feature_list_in.close();
  pair_list_in.close();
 */





  //if(shuffle==1)
  //{
	 // LOG(INFO)<<"Shuffling data";
	 // std::random_shuffle(pair_list.begin(), pair_list.end());
  //}
  // int num_pairs=pair_list.size();

  //LOG(INFO)<<"A total of "<<num_pairs<<" pairs";
 //  int num_triplet = feature_list1.size();  // the triplet size
  // Open leveldb
  leveldb::DB* db;
  leveldb::Options options;
  options.create_if_missing = true;
  options.error_if_exists = true;
  leveldb::Status status = leveldb::DB::Open(
      options, db_filename, &db);
  CHECK(status.ok()) << "Failed to open leveldb " << db_filename
      << ". Is it already existing?";


  const int kMaxKeyLength = 100;
  char key[kMaxKeyLength];
  std::string value;

  
  
  for (int triplet = 0; triplet < number; ++triplet) {
    caffe::Datum datum;
    datum.set_channels(3*feature_dim);  // one channel for each image in the pair
    datum.set_height(1);
    datum.set_width(1);
	//read_feature(triplet,original,similar,different, feature_dim, datum);
	//datum.add_float_data();
	for (int i = 0; i<dim; i++)
	{
		feature_list_in1 >> feature;
		datum.add_float_data(feature);
	}

	for (int i = 0; i < dim; i++)
	{
		feature_list_in2 >> feature;
		datum.add_float_data(feature);
	}

	for (int i =0; i<dim; i++)
	{
		feature_list_in3 >> feature;
		datum.add_float_data(feature);
	}
    datum.set_label(1);
    datum.SerializeToString(&value);
    _snprintf(key, kMaxKeyLength, "%08d", triplet);
    db->Put(leveldb::WriteOptions(), std::string(key), value);
	if (triplet % 10000 == 0)
		std::cout << "triplet:" << triplet<<"\n";
  }

  delete db;
  feature_list_in1.close();
  feature_list_in2.close();
  feature_list_in3.close();
}

int main(int argc, char** argv) {
	//argc = 7;
	//argv[1] = "D:\\users\\v-fulong\\17_HashCoding\\protox\\Original_cifar10_ip1.txt";
	//argv[2] = "D:\\users\\v-fulong\\17_HashCoding\\protox\\Similar_cifar10_ip1.txt";
	//argv[3] = "D:\\users\\v-fulong\\17_HashCoding\\protox\\Different_cifar10_ip1.txt";
	//argv[4] = "D:\\users\\v-fulong\\17_HashCoding\\Merge_ip1_ld";
	//argv[1] = "D:\\users\\v-fulong\\17_HashCoding\\TripletList_test\\original_ip1.txt";
	//argv[2] = "D:\\users\\v-fulong\\17_HashCoding\\TripletList_test\\similar_ip1.txt";
	//argv[3] = "D:\\users\\v-fulong\\17_HashCoding\\TripletList_test\\different_ip1.txt";
	//argv[4] = "D:\\users\\v-fulong\\17_HashCoding\\TripletList_test\\different_ip1.txt";
	//argv[5] = "10";
	//argv[1] = "D:\\users\\v-fulong\\17_HashCoding\\FeatureInput\\Original_cifar10_pool3_200w.txt";
	//argv[2] = "D:\\users\\v-fulong\\17_HashCoding\\FeatureInput\\Similar_cifar10_pool3_200w.txt";
	//argv[3] = "D:\\users\\v-fulong\\17_HashCoding\\FeatureInput\\Different_cifar10_pool3_200w.txt";
	//argv[4] = "D:\\users\\v-fulong\\17_HashCoding\\FeatureInput\\Merge_pool3_train_ld";
	//argv[5] = "10";
	//argv[6] = "2000000";
  if (argc != 7) {
    printf("This script converts the MNIST dataset to the leveldb format used\n"
           "by caffe to train a siamese network.\n"
           "Usage:\n"
		   "    convert_mnist_data feature_list1 feature_list2 feature_list3  output_db_file feature_dim number "
           "\n"
           );
  } else {
    google::InitGoogleLogging(argv[0]);
    int feature_dim=atoi(argv[5]);
	int number_ = atoi(argv[6]);
    convert_dataset(argv[1], argv[2], argv[3], argv[4],feature_dim,number_);
  }
  return 0;
}
