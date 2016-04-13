#include <opencv2/core/core.hpp>

#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

	template <typename Dtype>
	VecLabelDataLayer<Dtype>::~VecLabelDataLayer<Dtype>() {
		this->JoinPrefetchThread();
	}

	template <typename Dtype>
	void VecLabelDataLayer<Dtype>::VecLabelDataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		// Initialize DB
		db_.reset(db::GetDB(this->layer_param_.data_param().backend()));
		db_->Open(this->layer_param_.data_param().source(), db::READ);
		cursor_.reset(db_->NewCursor());

		// Check if we should randomly skip a few data points
		if (this->layer_param_.data_param().rand_skip()) {
			unsigned int skip = caffe_rng_rand() %
				this->layer_param_.data_param().rand_skip();
			LOG(INFO) << "Skipping first " << skip << " data points.";
			while (skip-- > 0) {
				cursor_->Next();
			}
		}
		// Read a data point, and use it to initialize the top blob.
		Datum datum;
		datum.ParseFromString(cursor_->value());

		bool force_color = this->layer_param_.data_param().force_encoded_color();
		if ((force_color && DecodeDatum(&datum, true)) ||
			DecodeDatumNative(&datum)) {
			LOG(INFO) << "Decoding Datum";
		}
		// image
		// modified by qing li to support inputing a video and make transformation on video level
		int crop_size = this->layer_param_.transform_param().crop_size();
		int crop_by_time_length = this->layer_param_.transform_param().crop_by_time_length();
		int real_height = datum.height();
		int real_width = datum.width();
		int real_channels = datum.channels();
		if (crop_size > 0)
		{
			real_height = real_width = crop_size;
		}
		if (crop_by_time_length > 0)
		{
			int time_unit = this->layer_param_.transform_param().time_unit();
			CHECK((datum.channels() % time_unit) == 0)
				<< "datum_channels is not divisible by time_unit";
			real_channels = crop_by_time_length*time_unit;
		}
		top[0]->Reshape(this->layer_param_.data_param().batch_size(),
			real_channels, real_height, real_width);
		this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
			real_channels, real_height, real_width);
		this->transformed_data_.Reshape(1, real_channels, real_height, real_width);

		//int crop_size = this->layer_param_.transform_param().crop_size();
		//if (crop_size > 0) {
		//  top[0]->Reshape(this->layer_param_.data_param().batch_size(),
		//      datum.channels(), crop_size, crop_size);
		//  this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
		//      datum.channels(), crop_size, crop_size);
		//  this->transformed_data_.Reshape(1, datum.channels(), crop_size, crop_size);
		//} else {
		//  top[0]->Reshape(
		//      this->layer_param_.data_param().batch_size(), datum.channels(),
		//      datum.height(), datum.width());
		//  this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
		//      datum.channels(), datum.height(), datum.width());
		//  this->transformed_data_.Reshape(1, datum.channels(),
		//    datum.height(), datum.width());
		//}
		LOG(INFO) << "output data size: " << top[0]->num() << ","
			<< top[0]->channels() << "," << top[0]->height() << ","
			<< top[0]->width();
		// label
		if (this->output_labels_) {
			int classnum = this->layer_param_.veclabel_data_param().class_num();
			vector<int> label_shape(classnum, this->layer_param_.data_param().batch_size());
			top[1]->Reshape(label_shape);
			this->prefetch_label_.Reshape(label_shape);
		}
	}

	// This function is used to create a thread that prefetches the data.
	template <typename Dtype>
	void VecLabelDataLayer<Dtype>::InternalThreadEntry() {
		CPUTimer batch_timer;
		batch_timer.Start();
		double read_time = 0;
		double trans_time = 0;
		int classnum = this->layer_param_.veclabel_data_param().class_num();
		CPUTimer timer;
		CHECK(this->prefetch_data_.count());
		CHECK(this->transformed_data_.count());

		// Reshape on single input batches for inputs of varying dimension.
		const int batch_size = this->layer_param_.data_param().batch_size();
		const int crop_size = this->layer_param_.transform_param().crop_size();
		const int crop_by_time_length = this->layer_param_.transform_param().crop_by_time_length();// added by qing li
		bool force_color = this->layer_param_.data_param().force_encoded_color();
		if (batch_size == 1 && crop_size == 0 && crop_by_time_length == 0) {
			Datum datum;
			datum.ParseFromString(cursor_->value());
			if (datum.encoded()) {
				if (force_color) {
					DecodeDatum(&datum, true);
				}
				else {
					DecodeDatumNative(&datum);
				}
			}
			this->prefetch_data_.Reshape(1, datum.channels(),
				datum.height(), datum.width());
			this->transformed_data_.Reshape(1, datum.channels(),
				datum.height(), datum.width());
		}

		Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
		Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

		if (this->output_labels_) {
			top_label = this->prefetch_label_.mutable_cpu_data();
		}
		for (int item_id = 0; item_id < batch_size; ++item_id) {
			timer.Start();
			// get a blob
			Datum datum;
			datum.ParseFromString(cursor_->value());

			cv::Mat cv_img;
			if (datum.encoded()) {
				if (force_color) {
					cv_img = DecodeDatumToCVMat(datum, true);
				}
				else {
					cv_img = DecodeDatumToCVMatNative(datum);
				}
				if (cv_img.channels() != this->transformed_data_.channels()) {
					LOG(WARNING) << "Your dataset contains encoded images with mixed "
						<< "channel sizes. Consider adding a 'force_color' flag to the "
						<< "model definition, or rebuild your dataset using "
						<< "convert_imageset.";
				}
			}
			read_time += timer.MicroSeconds();
			timer.Start();

			// Apply data transformations (mirror, scale, crop...)
			int offset = this->prefetch_data_.offset(item_id);
			this->transformed_data_.set_cpu_data(top_data + offset);
			if (datum.encoded()) {
				this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
			}
			else {
				this->data_transformer_->Transform(datum, &(this->transformed_data_));
			}
			//Make the label to the label vector add by Fuchen Long 1/1/2016
			if (this->output_labels_) {
				int displaylabel = datum.label();
				for (int label_pos = 0; label_pos < classnum; label_pos++)
				{
					if (label_pos == displaylabel)
					{
						top_label[item_id*classnum + label_pos] = 1;
					}
					else
					{
						top_label[item_id*classnum + label_pos] = 0;
					}		
				}
			}
			trans_time += timer.MicroSeconds();
			// go to the next iter
			cursor_->Next();
			if (!cursor_->valid()) {
				DLOG(INFO) << "Restarting data prefetching from start.";
				cursor_->SeekToFirst();
			}
		}
		batch_timer.Stop();
		DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
		DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
		DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
	}

	INSTANTIATE_CLASS(VecLabelDataLayer);
	REGISTER_LAYER_CLASS(VecLabelData);

}  // namespace caffe
