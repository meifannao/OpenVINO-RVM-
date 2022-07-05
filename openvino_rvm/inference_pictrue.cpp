#include <iostream>
#include <vector>
#include <string>
#include "opencv2/opencv.hpp"
#include "inference_engine.hpp"
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include "Tools.cpp"

using namespace InferenceEngine;
using namespace std;
using namespace cv;



void Inference_pictrue(string path)
{
	try
	{
		const int H = 180;
		const int W = 320;

		const float downsample_ratio = 0.5f;//无需改变
		SizeVector src_node_dims = { 1,3,H,W };
		std::vector<SizeVector> input_node_dims = initInputNodeDims(src_node_dims, downsample_ratio);
		float* r1o_data = nullptr;
		float* r2o_data = nullptr;
		float* r3o_data = nullptr;
		float* r4o_data = nullptr;

		string xmlPath = "automatic.xml";
		string binPath = "automatic.bin";

		Core ie;

		//1.加载模型
		CNNNetwork network = ie.ReadNetwork(xmlPath, binPath);

		//2.获取模型的输入结构并重新设置

		std::map<std::string, SizeVector> n_input;
		n_input.insert(std::make_pair("src", src_node_dims));
		n_input.insert(std::make_pair("1", input_node_dims.at(0)));
		n_input.insert(std::make_pair("2", input_node_dims.at(1)));
		n_input.insert(std::make_pair("3", input_node_dims.at(2)));
		n_input.insert(std::make_pair("4", input_node_dims.at(3)));
		network.reshape(n_input);
		lowLatency2(network);


		//3.设置输入数据
		ExecutableNetwork exec_network = ie.LoadNetwork(network, "CPU");
		InferRequest infer_request = exec_network.CreateInferRequest();

		Blob::Ptr srcBlob = infer_request.GetBlob("src");
		Blob::Ptr r1iBlob = infer_request.GetBlob("r1i");
		Blob::Ptr r2iBlob = infer_request.GetBlob("r2i");
		Blob::Ptr r3iBlob = infer_request.GetBlob("r3i");
		Blob::Ptr r4iBlob = infer_request.GetBlob("r4i");
		blobFromTensor(input_node_dims.at(0), r1iBlob);
		blobFromTensor(input_node_dims.at(1), r2iBlob);
		blobFromTensor(input_node_dims.at(2), r3iBlob);
		blobFromTensor(input_node_dims.at(3), r4iBlob);
		Blob::Ptr dstBlob, r1oBlob, r2oBlob, r4oBlob, r3oBlob;

		string CinPictruePath;
		CinPictruePath = path;
		
		Mat mat = imread(CinPictruePath);

		string CoutPicturePath;

		int len = CinPictruePath.size();

		for (int i = 0; i < len; i++)
		{
			if (CinPictruePath[i] != '.')
			{
				CoutPicturePath += CinPictruePath[i];
			}
			else
			{
				CoutPicturePath += "_result.png";
				break;
			}
		}

		cout << "输出路径：" << CoutPicturePath << endl;
		
		int src_H = 1080;
		int src_W = 1920;
		src_H = mat.cols;
		src_W = mat.rows;


		Mat bkg = Mat::zeros(Size(src_H, src_W), CV_8UC3);
	
		
		char strFPS[10];
		
		double ave = 0;

		Mat rzMat;
		cv::resize(mat, rzMat, Size(W, H));
		blobFromImage(rzMat, srcBlob);

		double t1 = cv::getTickCount();

		//4.模型推理
		infer_request.Infer();
		double time = ((double)cv::getTickCount() - t1) / cv::getTickFrequency();
		double fps = 1.0 / time;
		ave += fps;
		sprintf_s(strFPS, "%.2f", fps);
		std::string fpsString("FPS:");
		fpsString += strFPS;


		//5.模型输出
		dstBlob = infer_request.GetBlob("pha");
		float* dstvalue = dstBlob->buffer().as<float*>();
		r1oBlob = infer_request.GetBlob("r1o");
		r1o_data = r1oBlob->buffer().as<float*>();
		r2oBlob = infer_request.GetBlob("r2o");
		r2o_data = r2oBlob->buffer().as<float*>();
		r3oBlob = infer_request.GetBlob("r3o");
		r3o_data = r3oBlob->buffer().as<float*>();
		r4oBlob = infer_request.GetBlob("r4o");
		r4o_data = r4oBlob->buffer().as<float*>();

		Mat pha1(H, W, CV_32FC1, dstvalue);//pha即为模型输出的人像部分，后续的所有操作都可以基于pha进行处理
		Mat pha;
		cv::resize(pha1, pha, Size(src_H, src_W));
	
		Mat outVideo;


		float maxValue = *max_element(pha.begin<float>(), pha.end<float>());
		float minValue = *min_element(pha.begin<float>(), pha.end<float>());
		

		pha.convertTo(outVideo, CV_8U, 255.0 / (maxValue - minValue), -255.0 * minValue / (maxValue - minValue));
		
		imwrite(CoutPicturePath, outVideo);


	//	int key = waitKey(1);
		
			
			
		Mat ori_mat, resMat;
		int sz = 111;
		resize(pha, ori_mat, Size(src_W, src_H), 0, 0);
		replaceBackground(mat, bkg, ori_mat, resMat, sz);
		putText(resMat, fpsString, cv::Point(15, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
			
		cout << "FPS" << " " << ave << endl;
	}
	catch (const std::exception& e)
	{
		cout << e.what() << endl;
	}
}


