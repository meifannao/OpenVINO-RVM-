#include <iostream>
#include <vector>
#include <string>
#include "opencv2/opencv.hpp"
#include "inference_engine.hpp"
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include "inference_video.cpp"
#include "inference_pictrue.cpp"

using namespace InferenceEngine;
using namespace std;
using namespace cv;


bool Is_video(string test)
{
	int len = test.size();
	for (int i = 0; i < len; i++)
	{
		test[i] = tolower(test[i]);
	}

	if (test == "mp4" || test == "avi" || test == "dat" || test == "flv" || test == "wmv"
		|| test == "mpg" || test == "mpeg") return true;
	return false;
}


int main(int argc, char* argv[])
{
	try
	{
		argc++;
		
		string Inputpath = argv[1];


		//处理前后引号
		while (Inputpath.front() == 34) Inputpath.erase(Inputpath.begin());
		while (Inputpath.back() == 34) Inputpath.erase(Inputpath.end() - 1);

		//处理路径斜杠
		string tmp;
		int len = Inputpath.size();
		for (int i = 0; i < len; i++)
		{
			if (Inputpath[i] != 92)
			{
				tmp += Inputpath[i];
			}
			else if(i + 1 < len && Inputpath[i + 1] != 92)
			{
				tmp += "/";
			}
		}
		cout << tmp << endl;
		cout << "Running" << endl;
		if (argc < 2)
		{
			cout << "Please input video path" << endl;
		}
		else
		{
			string path = Inputpath;
			int k = path.find('.');
			if (k == string::npos)
			{
				cout << "Please input rigrh path" << endl;
			}
			else
			{
				string input_type = path.substr(k + 1);
				cout << input_type << endl;
				if (Is_video(input_type))
				{
					Inference_video(path);
				}
				else
				{
					Inference_pictrue(path);
				}
			}
		}
		cout << "Inference is Finish" << endl;
	}
	catch (const std::exception& e)
	{
		cout << e.what() << endl;
	}
}

