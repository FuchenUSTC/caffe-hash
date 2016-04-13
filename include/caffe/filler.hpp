// Fillers are random number generators that fills a blob using the specified
// algorithm. The expectation is that they are only going to be used during
// initialization time and will not involve any GPUs.

#ifndef CAFFE_FILLER_HPP
#define CAFFE_FILLER_HPP

#include <string>
#include <iostream>
#include <fstream>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

/// @brief Fills a Blob with constant or randomly-generated data.
template <typename Dtype>
class Filler {
 public:
  explicit Filler(const FillerParameter& param) : filler_param_(param) {}
  virtual ~Filler() {}
  virtual void Fill(Blob<Dtype>* blob) = 0;
  virtual void Fill_noisy(Blob<Dtype>* blob, string source) = 0;
 protected:
  FillerParameter filler_param_;
};  // class Filler


/// @brief Fills a Blob with constant values @f$ x = 0 @f$.
template <typename Dtype>
class ConstantFiller : public Filler<Dtype> {
 public:
  explicit ConstantFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    const int count = blob->count();
	const int num = blob->num();
    const Dtype value = this->filler_param_.value();
    CHECK(count);
    for (int i = 0; i < count; ++i) {
      data[i] = value;
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
  virtual void Fill_noisy(Blob<Dtype>* blob, string source){}
};

/// @brief Fills a Blob with uniformly distributed values @f$ x\sim U(a, b) @f$.
template <typename Dtype>
class UniformFiller : public Filler<Dtype> {
 public:
  explicit UniformFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    caffe_rng_uniform<Dtype>(blob->count(), Dtype(this->filler_param_.min()),
        Dtype(this->filler_param_.max()), blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
  virtual void Fill_noisy(Blob<Dtype>* blob, string source){}
};

/// @brief Fills a Blob with Gaussian-distributed values @f$ x = a @f$.
template <typename Dtype>
class GaussianFiller : public Filler<Dtype> {
 public:
  explicit GaussianFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    CHECK(blob->count());
    caffe_rng_gaussian<Dtype>(blob->count(), Dtype(this->filler_param_.mean()),
        Dtype(this->filler_param_.std()), blob->mutable_cpu_data());
    int sparse = this->filler_param_.sparse();
    CHECK_GE(sparse, -1);
    if (sparse >= 0) {
      // Sparse initialization is implemented for "weight" blobs; i.e. matrices.
      // These have num == channels == 1; width is number of inputs; height is
      // number of outputs.  The 'sparse' variable specifies the mean number
      // of non-zero input weights for a given output.
      CHECK_GE(blob->num_axes(), 1);
      const int num_outputs = blob->shape(0);
      Dtype non_zero_probability = Dtype(sparse) / Dtype(num_outputs);
      rand_vec_.reset(new SyncedMemory(blob->count() * sizeof(int)));
      int* mask = reinterpret_cast<int*>(rand_vec_->mutable_cpu_data());
      caffe_rng_bernoulli(blob->count(), non_zero_probability, mask);
      for (int i = 0; i < blob->count(); ++i) {
        data[i] *= mask[i];
      }
    }
  }
  virtual void Fill_noisy(Blob<Dtype>* blob, string source){}

 protected:
  shared_ptr<SyncedMemory> rand_vec_;
};

/** @brief Fills a Blob with values @f$ x \in [0, 1] @f$
 *         such that @f$ \forall i \sum_j x_{ij} = 1 @f$.
 */
template <typename Dtype>
class PositiveUnitballFiller : public Filler<Dtype> {
 public:
  explicit PositiveUnitballFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    DCHECK(blob->count());
    caffe_rng_uniform<Dtype>(blob->count(), 0, 1, blob->mutable_cpu_data());
    // We expect the filler to not be called very frequently, so we will
    // just use a simple implementation
    int dim = blob->count() / blob->num();
    CHECK(dim);
    for (int i = 0; i < blob->num(); ++i) {
      Dtype sum = 0;
      for (int j = 0; j < dim; ++j) {
        sum += data[i * dim + j];
      }
      for (int j = 0; j < dim; ++j) {
        data[i * dim + j] /= sum;
      }
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
  virtual void Fill_noisy(Blob<Dtype>* blob, string source){}
};

/**
 * @brief Fills a Blob with values @f$ x \sim U(-a, +a) @f$ where @f$ a @f$
 *        is set inversely proportional to the number of incoming nodes.
 *
 * A Filler based on the paper [Bengio and Glorot 2010]: Understanding
 * the difficulty of training deep feedforward neuralnetworks, but does not
 * use the fan_out value.
 *
 * It fills the incoming matrix by randomly sampling uniform data from
 * [-scale, scale] where scale = sqrt(3 / fan_in) where fan_in is the number
 * of input nodes. You should make sure the input blob has shape (num, a, b, c)
 * where a * b * c = fan_in.
 *
 * TODO(dox): make notation in above comment consistent with rest & use LaTeX.
 */
template <typename Dtype>
class XavierFiller : public Filler<Dtype> {
 public:
  explicit XavierFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    int fan_in = blob->count() / blob->num();
    Dtype scale = sqrt(Dtype(3) / fan_in);
    caffe_rng_uniform<Dtype>(blob->count(), -scale, scale,
        blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
  virtual void Fill_noisy(Blob<Dtype>* blob, string source){}
};

/// @brief filler used in RCSLayer.
template <typename Dtype>
class RCSFiller : public Filler<Dtype> {
public:
	explicit RCSFiller(const FillerParameter& param)
		: Filler<Dtype>(param) {}
	virtual void Fill(Blob<Dtype>* blob) {
		Dtype* data = blob->mutable_cpu_data();
		int count = blob->count();
		CHECK(count);
		data[0] = this->filler_param_.value();
		caffe_rng_gaussian<Dtype>(count-1, Dtype(this->filler_param_.mean()),
			Dtype(this->filler_param_.std()), &(data[1]));
	}
	virtual void Fill_noisy(Blob<Dtype>* blob, string source){}
};

/// @brief filler used in SSDLayer, not implemented.
template <typename Dtype>
class SSDFiller : public Filler<Dtype> {
public:
	explicit SSDFiller(const FillerParameter& param)
		: Filler<Dtype>(param) {}
	virtual void Fill(Blob<Dtype>* blob) {
		Dtype* data = blob->mutable_cpu_data();
		int count = blob->count();
		CHECK(count);
		data[0] = this->filler_param_.value();
		caffe_rng_gaussian<Dtype>(count-1, Dtype(this->filler_param_.mean()),
			Dtype(this->filler_param_.std()), &(data[1]));	
	}
	virtual void Fill_noisy(Blob<Dtype>* blob, string source){}
};

//@brief filler used in the NoisyLabel Learning added by Fuchen Long 4/1/2016
template<typename Dtype>
class IdentityFiller : public Filler<Dtype>{
public:
	explicit IdentityFiller(const FillerParameter &param)
	: Filler<Dtype>(param){}
	virtual void Fill(Blob<Dtype>* blob)
	{
		Dtype* data = blob->mutable_cpu_data();
		int count = blob->count();
		int num = blob->num(); // the output_num
		int dim = count / num; // the input_num
		CHECK(count);
		CHECK_EQ(num, dim) 
			<< "The Indentity Metrix should be the Squral Metrix. Input_num!=Output_num!";
		for (int i = 0; i < num; i++)
		{
			for (int j = 0; j < dim; j++)
			{
				if (i == j) data[i*dim + j] = 1;
				else data[i*dim + j] = 0;
			}
		}
		CHECK_EQ(this->filler_param_.sparse(), -1)
			<< "Sparsity not supported by this Filler.";
	}
	virtual void Fill_noisy(Blob<Dtype>* blob, string source){}
};

//@brief filler used in ther NoisyLabel Learning added by Fuchen Long 21/3/2016
//And this type is not returned by the GetFiller.
template<typename Dtype>
class NoisyTransFiller : public Filler<Dtype>{
public: 
	explicit NoisyTransFiller(const FillerParameter &param)
		:Filler<Dtype>(param){}
	void Fill_noisy(Blob<Dtype>* blob,string source)
	{
		std::ifstream noisymatrix(source, ios::in);
		Dtype* data = blob->mutable_cpu_data();
		int count = blob->count();
		int num = blob->num();  //the output number
		int dim = count / num;  //the input number
		CHECK(count);
		CHECK_EQ(num, dim)
			<< "The Noisy Trans Metrix should be the Squral Metrix. Input_num != Output_num!";
		Dtype temp;
		for (int i = 0; i < num; ++i){
			for (int j = 0; j < dim; ++j){
				noisymatrix >> temp;
				data[i*dim + j] = temp;
			}
		}
		CHECK_EQ(this->filler_param_.sparse(), -1)
			<< "Sparsity not supported by this Filler.";
	}
	void Fill(Blob<Dtype> * blob){}
};

	
/**
 * @brief Get a specific filler from the specification given in FillerParameter.
 *
 * Ideally this would be replaced by a factory pattern, but we will leave it
 * this way for now.
 */
template <typename Dtype>
Filler<Dtype>* GetFiller(const FillerParameter& param) {
  const std::string& type = param.type();
  if (type == "constant") {
    return new ConstantFiller<Dtype>(param);
  } else if (type == "gaussian") {
    return new GaussianFiller<Dtype>(param);
  } else if (type == "positive_unitball") {
    return new PositiveUnitballFiller<Dtype>(param);
  } else if (type == "uniform") {
    return new UniformFiller<Dtype>(param);
  } else if (type == "xavier") {
    return new XavierFiller<Dtype>(param);
  } else if (type == "rcs") {
  	return new RCSFiller<Dtype>(param);
  } else if (type == "ssd") {
  	return new SSDFiller<Dtype>(param);
  } else if (type == "identity"){
  	return new IdentityFiller<Dtype>(param);
  }else if (type == "noisytrans"){
	  return new NoisyTransFiller<Dtype>(param);
  }
	else {
    CHECK(false) << "Unknown filler name: " << param.type();
  }
  return (Filler<Dtype>*)(NULL);
}

}  // namespace caffe

#endif  // CAFFE_FILLER_HPP_
