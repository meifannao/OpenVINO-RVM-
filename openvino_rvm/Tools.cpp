
#pragma once
#include <iostream>
#include <vector>
#include <string>
#include "opencv2/opencv.hpp"
#include "inference_engine.hpp"
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

using namespace InferenceEngine;
using namespace std;
using namespace cv;

// 图像背景替换
void replaceBackground(Mat mSrc, Mat mBkg, Mat mMask, Mat & mDst, int sz)
{
	int wd = mSrc.cols;
	int ht = mSrc.rows;

	mDst = mSrc.clone();

	for (int i = 0; i < ht; i++)
	{
		uchar* dataSrc = mSrc.ptr<uchar>(i);
		uchar* dataBkg = mBkg.ptr<uchar>(i);
		uchar* dataDst = mDst.ptr<uchar>(i);
		float* dataMask = mMask.ptr<float>(i);

		for (int j = 0; j < wd; j++)
		{
			uchar BSrc = *(dataSrc + 3 * j);
			uchar GSrc = *(dataSrc + 3 * j + 1);
			uchar RSrc = *(dataSrc + 3 * j + 2);

			uchar BDst = *(dataBkg + 3 * j);
			uchar GDst = *(dataBkg + 3 * j + 1);
			uchar RDst = *(dataBkg + 3 * j + 2);

			float VMask = *(dataMask + j);

			*(dataDst + 3 * j) = VMask * BSrc + (1 - VMask) * BDst;
			*(dataDst + 3 * j + 1) = VMask * GSrc + (1 - VMask) * GDst;
			*(dataDst + 3 * j + 2) = VMask * RSrc + (1 - VMask) * RDst;

		}
	}
	return;
}

//对输入图像进行预处理
void blobFromImage(cv::Mat& img, InferenceEngine::Blob::Ptr& blob)
{
	int channels = 3;
	int img_h = img.rows;
	int img_w = img.cols;
	
	float* blob_data = blob->buffer().as<float*>();
	for (size_t c = 0; c < channels; c++)
	{
		for (size_t h = 0; h < img_h; h++)
		{
			for (size_t w = 0; w < img_w; w++)
			{
				blob_data[c * img_w * img_h + h * img_w + w] =
					(float)img.at<cv::Vec3b>(h, w)[c] / 255.;
			}
		}
	}
	//printf("blob_data = %.6f\n", *blob_data);
}


void blobFromTensor(SizeVector tensor, InferenceEngine::Blob::Ptr& blob)
{
	int channel = tensor.at(1);
	int height = tensor.at(2);
	int width = tensor.at(3);
	
	float* blob_data = blob->buffer().as<float*>();
	for (size_t c = 0; c < channel; c++)
	{
		for (size_t h = 0; h < height; h++)
		{
			for (size_t w = 0; w < width; w++)
			{
				blob_data[c * height * width + h * width + w] =
					0.0f;
			}
		}
	}
	//printf("blob_data = %.6f\n", *blob->buffer().as<float*>());
}

SizeVector calRSize(SizeVector& r_node_dims, size_t c)
{
	SizeVector rsize = { 1,c, 0,0 };
	int h, w;
	h = (r_node_dims.at(2) % 2 == 1) ? ((r_node_dims.at(2) + 1) / 2) : (r_node_dims.at(2) / 2);
	w = (r_node_dims.at(3) % 2 == 1) ? ((r_node_dims.at(3) + 1) / 2) : (r_node_dims.at(3) / 2);
	rsize.at(2) = h;
	rsize.at(3) = w;
	return rsize;
}

//根据H,W的值动态调整r1i、r2i..的值
std::vector<SizeVector> initInputNodeDims(SizeVector& src_node_dims, float downsample_ratio)
{
	SizeVector channels = { 16,20,40,64 };
	int down_h = src_node_dims.at(2) * downsample_ratio;
	int down_w = src_node_dims.at(3) * downsample_ratio;
	SizeVector src_dims = { 1,3,(size_t)down_h,(size_t)down_w };
	std::vector<SizeVector> input_node_dims;
	SizeVector r1_node_dims = calRSize(src_dims, channels.at(0));
	SizeVector r2_node_dims = calRSize(r1_node_dims, channels.at(1));
	SizeVector r3_node_dims = calRSize(r2_node_dims, channels.at(2));
	SizeVector r4_node_dims = calRSize(r3_node_dims, channels.at(3));
	input_node_dims = { r1_node_dims ,r2_node_dims ,r3_node_dims ,r4_node_dims };
	return input_node_dims;
}



int64_t value_size_of(const SizeVector& dims)
{
	if (dims.empty()) return 0;
	int64_t value_size = 1;
	for (const auto& size : dims) value_size *= size;
	return value_size;
}