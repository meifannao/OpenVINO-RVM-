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



void Inference_video(string path)
{
	try
	{
		//H,W为模型可接受的宽高，可以适当按照16：9进行调整。
		const int H = 180;
		const int W = 320;


		const float downsample_ratio = 0.5f;
		SizeVector src_node_dims = { 1,3,H,W };
		std::vector<SizeVector> input_node_dims = initInputNodeDims(src_node_dims, downsample_ratio);
		float* r1o_data = nullptr;
		float* r2o_data = nullptr;
		float* r3o_data = nullptr;
		float* r4o_data = nullptr;
		
		string xmlPath = "automatic.xml";
		string binPath = "automatic.bin";


		Core ie;

		//1.加载IR文件
		CNNNetwork network = ie.ReadNetwork(xmlPath, binPath);

		//cout << 1 << endl;

		//2.获取模型的输入结构并重新设置
		std::map<std::string, SizeVector> n_input;
		n_input.insert(std::make_pair("src", src_node_dims));
		n_input.insert(std::make_pair("r1i", input_node_dims.at(0)));
		n_input.insert(std::make_pair("r2i", input_node_dims.at(1)));
		n_input.insert(std::make_pair("r3i", input_node_dims.at(2)));
		n_input.insert(std::make_pair("r4i", input_node_dims.at(3)));
		network.reshape(n_input);
		lowLatency2(network);

		//cout << 2 << endl;

		
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

		//cout << 3 << endl;
		

		string CinVideoPath;
		CinVideoPath = path;
		VideoCapture video_capture(CinVideoPath);

		string CoutVideoPath, CoutVideopPath1;

		int len = CinVideoPath.size();

		for (int i = 0; i < len; i++)
		{
			if (CinVideoPath[i] != '.')
			{
				CoutVideoPath += CinVideoPath[i];
			}
			else
			{
				CoutVideopPath1 = CoutVideoPath;
				CoutVideopPath1 += "put.mp4";
				CoutVideoPath += "out.mp4";
				break;
			}
		}
		cout << "输出路径：" << CoutVideoPath << endl;
	
		int src_H = 1080;
		int src_W = 1920;
		src_W = video_capture.get(3);
		src_H = video_capture.get(4);
		

		//背景图
		Mat bkg = cv::Mat(src_H, src_W, CV_8UC3, cv::Scalar(255, 255, 255, 255));
		bkg.setTo(255);
		video_capture.set(cv::CAP_PROP_FRAME_WIDTH, src_W);
		video_capture.set(cv::CAP_PROP_FRAME_HEIGHT, src_H);
		Mat mat;
		int nFrame = 0;
		char strFPS[10];
		bool isNormal = false;
		bool isReplace = false;
		double ave = 0;
	
		cv::VideoWriter vw;
		int fps = video_capture.get(CAP_PROP_FPS);
		
		vw.open(CoutVideoPath, CAP_OPENCV_MJPEG, fps, cv::Size(src_W, src_H), true);

		while (1)
		{
			video_capture >> mat;
			
			if (mat.empty())
			{
				break;
			}

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
			
			//cout << 4 << endl;
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
			//cout << 5 << endl;
			//更新帧间信息
			float* r1i_data = r1iBlob->buffer().as<float*>();
			memcpy(r1i_data, r1o_data, value_size_of(input_node_dims[0]) * sizeof(float));
			float* r2i_data = r2iBlob->buffer().as<float*>();
			memcpy(r2i_data, r2o_data, value_size_of(input_node_dims[1]) * sizeof(float));
			float* r3i_data = r3iBlob->buffer().as<float*>();
			memcpy(r3i_data, r3o_data, value_size_of(input_node_dims[2]) * sizeof(float));
			float* r4i_data = r4iBlob->buffer().as<float*>();
			memcpy(r4i_data, r4o_data, value_size_of(input_node_dims[3]) * sizeof(float));

			Mat pha1(H, W, CV_32FC1, dstvalue);//pha
			Mat pha;
			cv::resize(pha1, pha, Size(src_W, src_H));
		
			imshow("pha", pha);
			Mat outVideo;


			float maxValue = *max_element(pha.begin<float>(), pha.end<float>());
			float minValue = *min_element(pha.begin<float>(), pha.end<float>());

			pha.convertTo(outVideo, CV_8U, 255.0 / (maxValue - minValue), -255.0 * minValue / (maxValue - minValue));
			vw << outVideo;


			int key = waitKey(1);
			if (isNormal)
			{
				putText(mat, fpsString, cv::Point(15, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
			}
			else
			{
				Mat ori_mat, resMat;
				int sz = 111;
				resize(pha, ori_mat, Size(src_W, src_H), 0, 0);
				replaceBackground(mat, bkg, ori_mat, resMat, sz);
			}
			nFrame++;
		}
		cout << nFrame << endl;
		vw.release();
		cout << "FPS" << " " << ave / nFrame << endl;
	}
	catch (const std::exception& e)
	{
		cout << e.what() << endl;
	}
}


