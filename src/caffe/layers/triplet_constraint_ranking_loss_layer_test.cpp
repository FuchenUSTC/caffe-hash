#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"
//test
#include <stdlib.h>
#include<iostream>
#include<ostream>
#include<istream>
//

//added by fuchen long for hash coding
namespace caffe {
	template <typename Dtype>
	void TripletConstraintRankingLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top
		)
	{
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		int codelength = this->layer_param_.triplet_constraint_ranking_loss_param().codelength();
		CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
		CHECK_EQ(bottom[1]->channels(), bottom[2]->channels());
		CHECK_EQ(bottom[0]->channels(), codelength);// the code length
		CHECK_EQ(bottom[1]->channels(), codelength);
		CHECK_EQ(bottom[2]->channels(), codelength);
		diff_.Reshape(bottom[0]->num(), 1, 1, 1);
		dist_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
		diff_sub_or_si.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1); // F-F+
		diff_sub_or_di.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1); // F-F-
		diff_pow_or_si.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1); // Pow (F-F+)
		diff_pow_or_di.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1); // Pow (F-F-)
		gradientTriplet.Reshape(1, bottom[0]->channels(), 1, 1);
		gradientConstraint.Reshape(1, bottom[0]->channels(), 1, 1);
		gradient.Reshape(1, bottom[0]->channels(), 1, 1);
	}



	template <typename Dtype>
	void TripletConstraintRankingLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top
		)
	{

		const int batchsize = bottom[0]->num();
		const int codelength = this->layer_param_.triplet_constraint_ranking_loss_param().codelength();
		int Dimv = batchsize*codelength;
		int MetricDimv = batchsize*batchsize;
		const Dtype* sub_or_si;
		const Dtype* sub_or_di;
		Dtype b = 2;
		Dtype Tripletlosstotal(0.0);
		Dtype Constraintlosstotal(0.0);
		Dtype Totalloss(0.0);
		Dtype Scale = this->layer_param_.triplet_constraint_ranking_loss_param().scale();


		//Test
		char * FileCode = "D:\\4-Cifar10-test-constraint\\DebugFile\\CodeFile.txt";
		std::ofstream CodeOut(FileCode, std::ios::out);
		for (int j = 0; j < 3; j++)
		{
			CodeOut << j + 1 << ": " << endl;
			for (int k = 0; k < batchsize; k++)
			{

				for (int i = 0; i < codelength; i++)
				{
					std::cout << bottom[j]->cpu_data()[i+k*codelength] << " ";
					CodeOut << bottom[j]->cpu_data()[i + k*codelength] << " ";
				}
				std::cout << endl;
				CodeOut << endl;
			}
		}
		//CodeOut.close();
		//

		//The triplet ranking loss
		caffe_sub(Dimv, bottom[0]->cpu_data(), bottom[1]->cpu_data(), diff_sub_or_si.mutable_cpu_data()); // F-F+
		caffe_sub(Dimv, bottom[0]->cpu_data(), bottom[2]->cpu_data(), diff_sub_or_di.mutable_cpu_data()); // F-F-
		caffe_powx(Dimv, diff_sub_or_si.cpu_data(), Dtype(2.0), diff_pow_or_si.mutable_cpu_data());		  //Pow
		caffe_powx(Dimv, diff_sub_or_di.cpu_data(), Dtype(2.0), diff_pow_or_di.mutable_cpu_data());       //Pow
		for (int n = 0; n < batchsize; n++)
		{
			sub_or_si = diff_pow_or_si.cpu_data() + diff_pow_or_si.offset(n);
			sub_or_di = diff_pow_or_di.cpu_data() + diff_pow_or_di.offset(n);
			Dtype result1 = 0;
			Dtype result2 = 0;
			result1 = caffe_cpu_asum(codelength, sub_or_si);
			result2 = caffe_cpu_asum(codelength, sub_or_di);
			Dtype loss(0.0);
			loss = std::max(Dtype(1.0) + result1 - result2, Dtype(0));// compute the loss
			diff_.mutable_cpu_data()[n] = loss; // save the loss[i]
		}
		for (int k = 0; k < batchsize; k++)
		{

			dist_sq_.mutable_cpu_data()[k] = diff_.cpu_data()[k];// save the loss[i] for BP
			Tripletlosstotal += dist_sq_.cpu_data()[k];
			//Test
			std::cout << diff_.cpu_data()[k]<<" ";
			CodeOut << diff_.cpu_data()[k] << " ";
			//
		}
		std::cout << endl;
		CodeOut << endl;

		Tripletlosstotal = Tripletlosstotal / static_cast<Dtype>(bottom[0]->num()); //get the average loss



		//The constraint loss
		Dtype *UnitMetric = new Dtype[MetricDimv];

		for (int i = 0; i < batchsize; i++)
		{

			for (int j = 0; j < batchsize; j++)
			{
				if (j == i) UnitMetric[j + i*batchsize] = Dtype(1.0);
				else UnitMetric[j + i*batchsize] = Dtype(0.0);
				//Test
				std::cout << UnitMetric[j + i*batchsize] << " ";
				CodeOut << UnitMetric[j + i*batchsize] << " ";
			}
			//Test
			std::cout << endl;
			CodeOut << endl;
		}
		
		Dtype *OriginalMUti=new Dtype[MetricDimv];
		Dtype *SimilarMUti=new Dtype[MetricDimv];
		Dtype *DifferentMUti=new Dtype[MetricDimv];
		Dtype *OriginalT=new Dtype[MetricDimv];
		Dtype *SimilarT=new Dtype[MetricDimv];
		Dtype *DifferentT=new Dtype[MetricDimv];
		Dtype OriginalResults;
		Dtype SimilarResults;
		Dtype DifferentResults;
		const Dtype margin(1.0);
		const Dtype zero(0.0);


		for (int i = 0; i < MetricDimv; i++)
		{
			OriginalMUti[i] = Dtype(0.0);
			SimilarMUti[i] = Dtype(0.0);
			DifferentMUti[i] = Dtype(0.0);
		}


		caffe_cpu_gemm(CblasNoTrans, CblasTrans, batchsize, batchsize, codelength, margin, bottom[0]->cpu_data(), bottom[0]->cpu_data(), zero, OriginalMUti);
		caffe_cpu_gemm(CblasNoTrans, CblasTrans, batchsize, batchsize, codelength, margin, bottom[1]->cpu_data(), bottom[1]->cpu_data(), zero, SimilarMUti);
		caffe_cpu_gemm(CblasNoTrans, CblasTrans, batchsize, batchsize, codelength, margin, bottom[2]->cpu_data(), bottom[2]->cpu_data(), zero, DifferentMUti);

		//Test
		CodeOut << "ATA" << " " << endl;
		for (int i = 0; i < batchsize; i++)
		{
			for (int j = 0; j < batchsize; j++)
			{
				std::cout << OriginalMUti[j + i*batchsize] << " ";
				CodeOut << OriginalMUti[j + i*batchsize] << " ";
			}
			std::cout << endl;
			CodeOut << endl;
		}
		CodeOut << endl;
		//

		caffe_sub(MetricDimv, OriginalMUti, UnitMetric, OriginalMUti);
		caffe_sub(MetricDimv, SimilarMUti, UnitMetric, SimilarMUti);
		caffe_sub(MetricDimv, DifferentMUti, UnitMetric, DifferentMUti);

		//Test
		CodeOut << "ATA-I" << " " << endl;
		for (int i = 0; i < batchsize; i++)
		{
			for (int j = 0; j < batchsize; j++)
			{
				std::cout << OriginalMUti[j + i*batchsize] << " ";
				CodeOut << OriginalMUti[j + i*batchsize] << " ";
			}
			std::cout << endl;
			CodeOut << endl;
		}
		CodeOut << endl;

		//

		caffe_powx(MetricDimv, OriginalMUti, b, OriginalT);
		caffe_powx(MetricDimv, SimilarMUti, b, SimilarT);
		caffe_powx(MetricDimv, DifferentMUti, b, DifferentT);

		//Test
		CodeOut << "(ATA-I)2" << " " << endl;
		for (int i = 0; i < batchsize; i++)
		{
			for (int j = 0; j < batchsize; j++)
			{
				std::cout << OriginalT[j + i*batchsize] << " ";
				CodeOut << OriginalT[j + i*batchsize] << " ";
			}
			std::cout << endl;
			CodeOut << endl;
		}
		CodeOut << endl;
		//

		OriginalResults = caffe_cpu_asum(MetricDimv, OriginalT);
		SimilarResults = caffe_cpu_asum(MetricDimv, SimilarT);
		DifferentResults = caffe_cpu_asum(MetricDimv, DifferentT);

		//Test
		CodeOut << "Sum(ATA-I)2: " << OriginalResults << endl;
		CodeOut << endl;
		//
		CodeOut.close();
		Constraintlosstotal = (OriginalResults + SimilarResults + DifferentResults) / (Scale*static_cast<Dtype>(bottom[0]->num()));

		//Fusion the two lossresults
		Dtype lamda = this->layer_param_.triplet_constraint_ranking_loss_param().lamda();
		Totalloss = (lamda)*Tripletlosstotal + (Dtype(1.0) - lamda)*Constraintlosstotal;
		top[0]->mutable_cpu_data()[0] = Totalloss;

		//delete the Dtype*

		delete[] OriginalMUti;
		delete[] SimilarMUti;
		delete[] DifferentMUti; 
		delete[] OriginalT;
		delete[] SimilarT;
		delete[] DifferentT;
		delete[] UnitMetric;

	}


	template <typename Dtype>
	void TripletConstraintRankingLossLayer<Dtype>::Backward_cpu(
		const vector<Blob<Dtype>*>& top, const vector<bool> &propagate_down,
		const vector<Blob<Dtype>*>& bottom)
	{
		const int codelength = this->layer_param_.triplet_constraint_ranking_loss_param().codelength();
		int batchsize = bottom[0]->num();
		int Dimv = codelength*batchsize;
		int MetricDimv = batchsize*batchsize;
		Dtype lamda = this->layer_param_.triplet_constraint_ranking_loss_param().lamda();
		const Dtype* orignalcode;
		const Dtype* similarcode;
		const Dtype* diffrcode;
		Dtype Scale = this->layer_param_.triplet_constraint_ranking_loss_param().scale();
		//Dtype namda(0.0);
		Dtype *UnitMetric = new Dtype[MetricDimv];
		for (int i = 0; i < batchsize; i++)
		{
			for (int j = 0; j < batchsize; j++)
			{
				if (j == i) UnitMetric[j + i*batchsize] = Dtype(1.0);
				else UnitMetric[j + i*batchsize] = Dtype(0.0);
			}
		}
		Dtype *OriginalMUti = new Dtype[MetricDimv];
		Dtype *SimilarMUti = new Dtype[MetricDimv];
		Dtype *DifferentMUti = new Dtype[MetricDimv];
		Dtype *OriginalResults = new Dtype[Dimv];
		Dtype *SimilarResults = new Dtype[Dimv];
		Dtype *DifferentResults = new Dtype[Dimv];
		Dtype *Originalsub=new Dtype[MetricDimv];
		Dtype *Similarsub =new  Dtype[MetricDimv];
		Dtype *Differentsub =new  Dtype[MetricDimv];
		for (int i = 0; i < MetricDimv; i++)
		{
			OriginalMUti[i] = Dtype(0.0);
			SimilarMUti[i] = Dtype(0.0);
			DifferentMUti[i] = Dtype(0.0);
		}

		caffe_cpu_gemm(CblasNoTrans, CblasTrans, batchsize, batchsize, codelength, Dtype(1.0), bottom[0]->cpu_data(), bottom[0]->cpu_data(), Dtype(0.0), OriginalMUti);
		caffe_cpu_gemm(CblasNoTrans, CblasTrans, batchsize, batchsize, codelength, Dtype(1.0), bottom[1]->cpu_data(), bottom[1]->cpu_data(), Dtype(0.0), SimilarMUti);
		caffe_cpu_gemm(CblasNoTrans, CblasTrans, batchsize, batchsize, codelength, Dtype(1.0), bottom[2]->cpu_data(), bottom[2]->cpu_data(), Dtype(0.0), DifferentMUti);

		caffe_sub(MetricDimv, OriginalMUti, UnitMetric, Originalsub);
		caffe_sub(MetricDimv, SimilarMUti, UnitMetric, Similarsub);
		caffe_sub(MetricDimv, DifferentMUti, UnitMetric, Differentsub);

		caffe_cpu_gemm(CblasTrans, CblasNoTrans, codelength, batchsize, batchsize, Dtype(1.0), bottom[0]->cpu_data(), Originalsub, Dtype(0.0), OriginalResults);
		caffe_cpu_gemm(CblasTrans, CblasNoTrans, codelength, batchsize, batchsize, Dtype(1.0), bottom[1]->cpu_data(), Similarsub, Dtype(0.0), SimilarResults);
		caffe_cpu_gemm(CblasTrans, CblasNoTrans, codelength, batchsize, batchsize, Dtype(1.0), bottom[2]->cpu_data(), Differentsub, Dtype(0.0), DifferentResults);



		if (propagate_down[0]) {
			for (int i = 0; i < 3; ++i) {// for each stream need to get a loss

				int num = bottom[i]->num(); // get the layers' batchsize
				int channels = bottom[i]->channels();// get the layers' channels, channels==codelength
				for (int j = 0; j < num; ++j) // for each batch
				{
					Dtype* bout = bottom[i]->mutable_cpu_diff();// get the 3 bottoms' address, the i th bottom's address
					orignalcode = bottom[0]->cpu_data() + bottom[0]->offset(j);// get  the original image hash approximate code
					similarcode = bottom[1]->cpu_data() + bottom[1]->offset(j);// get the similar image hash approximate code
					diffrcode = bottom[2]->cpu_data() + bottom[2]->offset(j);//get the different image hash approximate code
					if (i == 0)// for the original bottom layer
					{
						if (dist_sq_.cpu_data()[j]>Dtype(0.0))//if the j th batch's loss > 0
						{
							caffe_sub(codelength, diffrcode, similarcode, gradientTriplet.mutable_cpu_data());// the distance of F- and F+
							caffe_scal(codelength, Dtype(2)*(lamda) / Dtype(num), gradientTriplet.mutable_cpu_data());// scale the 2/num
							//Test
							for (int x = 0; x < codelength; x++)
							{
								std::cout << gradientTriplet.cpu_data()[x] << " ";
							}
							std::cout << endl;
							//
						}
						else
							caffe_sub(codelength, diffrcode, diffrcode, gradientTriplet.mutable_cpu_data());// if the j th batch's loss <=0 ,return 0 vector
						//get the constraint gradient
						for (int k = 0; k < codelength; k++)
						{
							gradientConstraint.mutable_cpu_data()[k] = OriginalResults[j + k*batchsize];
							//Test
							std::cout << OriginalResults[j + k*batchsize] << endl;
							//
						}
						std::cout << "----------------------------------------------------" << endl;
						caffe_scal(codelength, Dtype(4)*(Dtype(1.0)-lamda) /(Scale*Dtype(num)), gradientConstraint.mutable_cpu_data());
						caffe_add(codelength, gradientTriplet.cpu_data(), gradientConstraint.cpu_data(), gradient.mutable_cpu_data());
						//Test
						for (int l = 0; l < codelength; l++)
						{
							std::cout << gradient.cpu_data()[l] << " ";
						}
						std::cout << endl;
						//

					}
					if (i == 1)// for the similar bottom layer
					{
						if (dist_sq_.cpu_data()[j] > Dtype(0.0))// if the j th batch's loss > 0
						{
							caffe_sub(codelength, similarcode, orignalcode, gradientTriplet.mutable_cpu_data());// the distance of F+ and F
							caffe_scal(codelength, Dtype(2)*(lamda) / Dtype(num), gradientTriplet.mutable_cpu_data());//scale the 2/num
							//Test
							for (int x = 0; x < codelength; x++)
							{
								std::cout << gradientTriplet.cpu_data()[x] << " ";
							}
							std::cout << endl;
							//

						}
						else
							caffe_sub(codelength, diffrcode, diffrcode, gradientTriplet.mutable_cpu_data());// if the j th batch's loss <=0, return 0 vector
						//get the constraint gradient
						for (int k = 0; k < codelength; k++)
						{
							gradientConstraint.mutable_cpu_data()[k] = SimilarResults[j + k*batchsize];
							//Test
							std::cout << SimilarResults[j + k*batchsize] << endl;
							//
						}
						std::cout << "-------------------------------------------------" << endl;
						caffe_scal(codelength, Dtype(4)*(Dtype(1.0) - lamda) /(Scale*Dtype(num)), gradientConstraint.mutable_cpu_data());
						caffe_add(codelength, gradientTriplet.cpu_data(), gradientConstraint.cpu_data(), gradient.mutable_cpu_data());
					}
					if (i == 2)// for the different bottom layer
					{
						if (dist_sq_.cpu_data()[j] > Dtype(0.0))// if the j th batch's loss > 0
						{
							caffe_sub(codelength, orignalcode, diffrcode, gradientTriplet.mutable_cpu_data()); // the distance of F and F-
							caffe_scal(codelength, Dtype(2)*(lamda) / Dtype(num), gradientTriplet.mutable_cpu_data());//scale the 2/num
							//Test
							for (int x = 0; x < codelength; x++)
							{
								std::cout << gradientTriplet.cpu_data()[x] << " ";
							}
							std::cout << endl;
							//
						}
						else
							caffe_sub(codelength, diffrcode, diffrcode, gradientTriplet.mutable_cpu_data());// if the j th batch's loss =0 ,return 0 vector
						//get the constraint gradient
						for (int k = 0; k < codelength; k++)
						{
							gradientConstraint.mutable_cpu_data()[k] = DifferentResults[j + k*batchsize];
							//Test
							std::cout << DifferentResults[j + k*batchsize] << endl;
							//
						}
						std::cout << "---------------------------------------------------" << endl;
						caffe_scal(codelength, Dtype(4)*(Dtype(1.0) - lamda) / (Scale*Dtype(num)), gradientConstraint.mutable_cpu_data());
						caffe_add(codelength, gradientTriplet.cpu_data(), gradientConstraint.cpu_data(), gradient.mutable_cpu_data());
					}
					//LOG(INFO) << "TEST 1\n";
					caffe_copy(channels, gradient.cpu_data(), bout + (j*channels));//return the BP vector to the j th batch's bottom
					//LOG(INFO) << "channels:" << channels;
					//LOG(INFO) << "gradient:" <<gradient[0] <<","<< gradient[1] <<","<< gradient[2]<<"," << gradient[3];
				}
			}
		}

		//delete the Dtype*
		delete[] OriginalMUti;
		delete[] SimilarMUti;
		delete[] DifferentMUti;
		delete[] OriginalResults;
		delete[] SimilarResults;
		delete[] DifferentResults;
		delete[] Originalsub;
		delete[] Similarsub;
		delete[] Differentsub;
		delete[] UnitMetric;
 	}

	INSTANTIATE_CLASS(TripletConstraintRankingLossLayer);
	REGISTER_LAYER_CLASS(TripletConstraintRankingLoss);


}