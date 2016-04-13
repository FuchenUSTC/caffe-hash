//added by fuchen long 2015-8-24
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

void convert_dataset(const char* feature_list_filename, const char*maping_list_filename, 
        const char* db_filename, int feature_dim,int numF,int numTriplet ) {
  // Open files
  std::ifstream feature_list_in1(feature_list_filename, std::ios::in); // The feature list
  std::ifstream maping_list_in2(maping_list_filename, std::ios::in);// The maping filename
  //std::ifstream feature_list_in3(feature_list_filename3, std::ios::in);// differernt
  //std::ifstream pair_list_in(pair_list_filename, std::ios::in);
  CHECK(feature_list_in1) << "Unable to open file " << feature_list_filename; // added by fuchen long
  CHECK(maping_list_in2) << "Unable to open file " << maping_list_filename;
  //CHECK(feature_list_in3) << "Unable to open file " << feature_list_filename3;
 // CHECK(pair_list_in) << "Unable to open file " << pair_list_filename;
  
  //read metadata
  //std::vector<std::string> feature_list1;
  //std::vector<std::string> feature_list2;
  //std::vector<std::string> feature_list3;
   int  dim = feature_dim;
   int numberFeature= numF;
   int numberTriplet = numTriplet;
   float feature;


   //get the feature
   float **TotalFeature = new float *[numberFeature];
   for (int i = 0; i < numberFeature; i++)
   {
	   TotalFeature[i] = new float[dim];
   }

   for (int i = 0; i < numberFeature; i++)
   {
	   for (int k = 0; k < dim; k++)
		   feature_list_in1 >> TotalFeature[i][k];
	   if (i % 1000 == 0)
		   std::cout << "have get the " << i << " features \n";
   }
   std::cout << "have get the feature\n";
   //get the maping
   int  **TripletMaping = new int *[numberTriplet];
   for (int i = 0; i < numberTriplet; i++)
   {
	   TripletMaping[i] = new int[3];
   }
   for (int i = 0; i < numberTriplet; i++)
   {
	   for (int k = 0; k < 3; k++)
		   maping_list_in2 >> TripletMaping[i][k];
	   if (i % 10000 == 0)
		   std::cout << "have get the " << i << " mapings \n";
   }

   std::cout << "The feature check " << TotalFeature[55][14]<< " "<< TotalFeature[209][13] << " \n";
   std::cout << "The maping check " << TripletMaping[15][1] << " " << TripletMaping[123][2] << " \n";
   feature_list_in1.close();
   maping_list_in2.close();

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
  options.write_buffer_size = 256 * 1024 * 1024;
  leveldb::Status status = leveldb::DB::Open(
      options, db_filename, &db);
  CHECK(status.ok()) << "Failed to open leveldb " << db_filename
      << ". Is it already existing?";


  const int kMaxKeyLength = 100;
  char key[kMaxKeyLength];
  std::string value;
  int index;
  
  
  for (int triplet = 0; triplet < numberTriplet; ++triplet) {
    caffe::Datum datum;
    datum.set_channels(3*feature_dim);  // one channel for each image in the pair
    datum.set_height(1);
    datum.set_width(1);
	//read_feature(triplet,original,similar,different, feature_dim, datum);
	//datum.add_float_data();

	index = TripletMaping[triplet][0] - 1;
	for (int i = 0; i<dim; i++)
	{
		datum.add_float_data(TotalFeature[index][i]);
	}

	index = TripletMaping[triplet][1] - 1;
	for (int i = 0; i < dim; i++)
	{		
		datum.add_float_data(TotalFeature[index][i]);
	}

	index = TripletMaping[triplet][2] - 1;
	for (int i =0; i<dim; i++)
	{
		datum.add_float_data(TotalFeature[index][i]);
	}
    datum.set_label(1);
    datum.SerializeToString(&value);
    _snprintf(key, kMaxKeyLength, "%08d", triplet);
    db->Put(leveldb::WriteOptions(), std::string(key), value);
	if (triplet % 1000 == 0)
		std::cout << "triplet:" << triplet<<"\n";
  }

  //delelt the db and others

  for (int i = 0; i < numberFeature; i++)
  {
	  delete[] TotalFeature[i];
  }
  delete[] TotalFeature;
  
  for (int i = 0; i < numberTriplet; i++)
  {
	  delete[] TripletMaping[i];
  }
  delete[] TripletMaping;
  delete db;
  //feature_list_in1.close();
  //feature_list_in2.close();
  //feature_list_in3.close();
}

int main(int argc, char** argv) {
	argc = 7;
	argv[1] = "D:\\17-MutilabelNusWide\\DataSet\\New_Nus_Sample\\Sample_List\\TestAndTrain_Label\\TripletLabel\\Testlabel.txt";
	argv[2] = "D:\\17-MutilabelNusWide\\DataSet\\New_Nus_Sample\\Sample_List\\TestAndTrain_Label\\TripletLabel\\Val_map.txt";
	argv[3] = "D:\\17-MutilabelNusWide\\DataSet\\New_Nus_Sample\\Sample_List\\TestAndTrain_Label\\TripletLabel\\Nus_ValLabel_ld";
	argv[4] = "21";
	argv[5] = "2100";
	argv[6] = "500";
  if (argc != 7) {
    printf("This script converts the MNIST dataset to the leveldb format used\n"
           "by caffe to train a siamese network.\n"
           "Usage:\n"
		   "    convert_triplet_feature feature_list triplet_list output_db_file feature_dim number_feature number_triplet "
           "\n"
           );
  } else {
    google::InitGoogleLogging(argv[0]);
	int feature_dim = atoi(argv[4]);
	int number_F = atoi(argv[5]);
	int number_T = atoi(argv[6]);
	std::cout << feature_dim << std::endl;
    convert_dataset(argv[1], argv[2], argv[3],feature_dim,number_F,number_T);
  }
  return 0;
}
