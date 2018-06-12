#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

//������·�����У�
#define  FILEPATH "E:\\VS_Programming\\OpenCV Program\\openCV-haar-boosted\\ObjectDection\\HOG-SVM\\opencv_HOG_SVM\\Pedestrians64x128\\"

///////////////////////////////////HOG+SVMʶ��ʽ2///////////////////////////////////////////////////  
void Train()
{
	////////////////////////////////����ѵ������ͼƬ·�������/////////////////////////
	//ͼ��·�������
	vector<string> imagePath;
	vector<int> imageClass;
	int numberOfLine = 0;
	string buffer;
	ifstream trainingData(string(FILEPATH) + "TrainData.txt");

	unsigned long n;
	while (!trainingData.eof())
	{
		getline(trainingData, buffer);
		if (!buffer.empty())
		{
			++numberOfLine;
			if (numberOfLine % 2 == 0)
			{
				//��ȡ�������
				imageClass.push_back(atoi(buffer.c_str()));
			}
			else
			{
				//��ȡͼ��·��
				imagePath.push_back(buffer);
			}
		}
	}

	//�ر��ļ�  
	trainingData.close();

	////////////////////////////////��ȡ������HOG����///////////////////////////////////
	//����������������
	int numberOfSample = numberOfLine / 2;
	Mat featureVectorOfSample(numberOfSample, 3780, CV_32FC1);  //������ÿ��Ϊһ������
	Mat classOfSample(numberOfSample, 1, CV_32SC1);				//���������

	Mat convertedImage;
	Mat trainImage;

	// ����HOG����
	for (vector<string>::size_type i = 0; i <= imagePath.size() - 1; ++i)
	{
		//����ͼƬ
		Mat src = imread(imagePath[i], -1);
		if (src.empty())
		{
			std::cout << "can not load the image:" << imagePath[i] << endl;
			continue;
		}

		// ��һ��
		resize(src, trainImage, Size(64, 128));

		// ��ȡHOG����
		HOGDescriptor hog(cvSize(64, 128), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
		vector<float> descriptors;
		double time1 = getTickCount();
		hog.compute(trainImage, descriptors);//����������ü�ⴰ�ڲ��������ͼƬ��С����64��128����������winStride
		double time2 = getTickCount();
		double elapse_ms = (time2 - time1)*1000 / getTickFrequency();   //���صĵ�λ����
		//std::cout << "HOG dimensions:" << descriptors.size() << endl;
		//std::cout << "Compute time:" << elapse_ms << endl;
		std::cout << "HOG Pic:" << i << "--->" << elapse_ms << endl;

		//���浽��������������
		for (vector<float>::size_type j = 0; j <= descriptors.size() - 1; ++j)
		{
			featureVectorOfSample.at<float>(i, j) = descriptors[j];
		}

		//�������������
		//!!ע���������һ��Ҫ��int ���͵�
		classOfSample.at<int>(i, 0) = imageClass[i];
	}


	///////////////////////////////////ʹ��SVM������ѵ��/////////////////////////////
	//���ò�����ע��Ptr��ʹ��
	Ptr<ml::SVM> svm = ml::SVM::create();
	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::LINEAR); //ע�����ʹ������SVM����ѵ������ΪHogDescriptor��⺯��ֻ֧�����Լ�⣡����
	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER, 1000, FLT_EPSILON));

	//ʹ��SVMѧϰ         
	svm->train(featureVectorOfSample, ml::ROW_SAMPLE, classOfSample);

	//���������(���������SVM�Ĳ�����֧������,����rho)
	svm->save(string(FILEPATH) + "Classifier.xml");

	/*
	SVMѵ����ɺ�õ���XML�ļ����棬��һ�����飬����support vector������һ�����飬����alpha,��һ��������������rho;
	��alpha����ͬsupport vector��ˣ�ע�⣬alpha*supportVector,���õ�һ������������������ǰ�����-1��֮���ٸ���������������һ��Ԫ��rho��
	��ˣ���õ���һ�������������ø÷�������ֱ���滻opencv�����˼��Ĭ�ϵ��Ǹ���������cv::HOGDescriptor::setSVMDetector()����
	*/
	//��ȡ֧��������������Ĭ����CV_32F

	Mat supportVector = svm->getSupportVectors();//

												 //��ȡalpha��rho
	Mat alpha;//ÿ��֧��������Ӧ�Ĳ�����(�������ճ���)��Ĭ��alpha��float64��
	Mat svIndex;//֧���������ڵ�����
	float rho = svm->getDecisionFunction(0, alpha, svIndex);

	//ת������:����һ��Ҫע�⣬��Ҫת��Ϊ32��
	Mat alpha2;
	alpha.convertTo(alpha2, CV_32FC1);

	//������������������
	Mat result(1, 3780, CV_32FC1);
	result = alpha2*supportVector;

	//����-1������Ϊʲô�����-1��
	//ע����Ϊsvm.predictʹ�õ���alpha*sv*another-rho�����Ϊ���Ļ�����Ϊ������������HOG�ļ�⺯���У�ʹ��rho+alpha*sv*another(anotherΪ-1)
	for (int i = 0; i < 3780; ++i)
		result.at<float>(0, i) *= -1;

	//�����������浽�ļ�������HOGʶ��
	//��������������б����Ĳ���(��)��HOG����ֱ��ʹ�øò�������ʶ��
	FILE *fp = fopen((string(FILEPATH) + "HOG_SVM.txt").c_str(), "wb");
	for (int i = 0; i < 3780; i++)
	{
		fprintf(fp, "%f \n", result.at<float>(0, i));
	}
	fprintf(fp, "%f", rho);

	fclose(fp);
}

// ʹ��ѵ���õķ�����ʶ��
void Detect()
{
	Mat img;
	FILE* f = 0;
	char _filename[1024];

	// ��ȡ����ͼƬ�ļ�·��
	f = fopen((string(FILEPATH) + "TestData.txt").c_str(), "rt");
	if (!f)
	{
		fprintf(stderr, "ERROR: the specified file could not be loaded\n");
		return;
	}

	//����ѵ���õ��б����Ĳ���(ע�⣬��svm->save����ķ�������ͬ)
	vector<float> detector;
	ifstream fileIn(string(FILEPATH) + "HOG_SVM.txt", ios::in);
	float val = 0.0f;
	while (!fileIn.eof())
	{
		fileIn >> val;
		detector.push_back(val);
	}
	fileIn.close();

	//����HOG
	HOGDescriptor hog;
	//hog.setSVMDetector(detector);// ʹ���Լ�ѵ���ķ�����
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());//����ֱ��ʹ��05 CVPR��ѵ���õķ�����,�����Ͳ���Train()���������

	namedWindow("people detector", 1);

	// ���ͼƬ
	for (;;)
	{
		// ��ȡ�ļ���
		char* filename = _filename;
		if (f)
		{
			if (!fgets(filename, (int)sizeof(_filename) - 2, f))
				break;
			//while(*filename && isspace(*filename))
			//  ++filename;
			if (filename[0] == '#')
				continue;

			//ȥ���ո�
			int l = (int)strlen(filename);
			while (l > 0 && isspace(filename[l - 1]))
				--l;
			filename[l] = '\0';
			img = imread(filename);
		}
		printf("%s:\n", filename);
		if (!img.data)
			continue;

		fflush(stdout);
		vector<Rect> found, found_filtered;
		double t = (double)getTickCount();
		// run the detector with default parameters. to get a higher hit-rate
		// (and more false alarms, respectively), decrease the hitThreshold and
		// groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
		//��߶ȼ��
		hog.detectMultiScale(img, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
		t = (double)getTickCount() - t;
		printf("detection time = %gms\n", t*1000. / cv::getTickFrequency());
		size_t i, j;

		//ȥ���ռ��о������������ϵ�����򣬱������
		for (i = 0; i < found.size(); i++)
		{
			Rect r = found[i];
			for (j = 0; j < found.size(); j++)
				if (j != i && (r & found[j]) == r)
					break;
			if (j == found.size())
				found_filtered.push_back(r);
		}

		// �ʵ���С����
		for (i = 0; i < found_filtered.size(); i++)
		{
			Rect r = found_filtered[i];
			// the HOG detector returns slightly larger rectangles than the real objects.
			// so we slightly shrink the rectangles to get a nicer output.
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.07);
			r.height = cvRound(r.height*0.8);
			rectangle(img, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
		}

		imshow("people detector", img);
		int c = waitKey(0) & 255;
		if (c == 'q' || c == 'Q' || !f)
			break;
	}
	if (f)
		fclose(f);
	return;


}

void HOG_SVM2()
{
	//���ʹ��05 CVPR�ṩ��Ĭ�Ϸ�����������ҪTrain(),ֱ��ʹ��Detect���ͼƬ
	Train();
	Detect();
}

///////////////////////////////////HOG+SVMʶ��ʽ1///////////////////////////////////////////////////
void HOG_SVM1()
{
	////////////////////////////////����ѵ������ͼƬ·�������///////////////////////////////////////////////////
	//ͼ��·�������
	vector<string> imagePath;
	vector<int> imageClass;
	int numberOfLine = 0;
	string buffer;
	ifstream trainingData(string(FILEPATH) + "TrainData.txt", ios::in);
	unsigned long n;

	while (!trainingData.eof())
	{
		getline(trainingData, buffer);
		if (!buffer.empty())
		{
			++numberOfLine;
			if (numberOfLine % 2 == 0)
			{
				//��ȡ�������
				imageClass.push_back(atoi(buffer.c_str()));
			}
			else
			{
				//��ȡͼ��·��
				imagePath.push_back(buffer);
			}
		}
	}
	trainingData.close();


	////////////////////////////////��ȡ������HOG����///////////////////////////////////////////////////
	//����������������
	int numberOfSample = numberOfLine / 2;
	Mat featureVectorOfSample(numberOfSample, 3780, CV_32FC1);//������ÿ��Ϊһ������

															  //���������
	Mat classOfSample(numberOfSample, 1, CV_32SC1);

	//��ʼ����ѵ��������HOG����
	for (vector<string>::size_type i = 0; i <= imagePath.size() - 1; ++i)
	{
		//����ͼƬ
		Mat src = imread(imagePath[i], -1);
		if (src.empty())
		{
			cout << "can not load the image:" << imagePath[i] << endl;
			continue;
		}
		cout << "processing" << imagePath[i] << endl;

		//����
		Mat trainImage;
		resize(src, trainImage, Size(64, 128));

		//��ȡHOG����
		HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
		vector<float> descriptors;
		hog.compute(trainImage, descriptors);//����������ü�ⴰ�ڲ��������ͼƬ��С����64��128����������winStride
		cout << "HOG dimensions:" << descriptors.size() << endl;

		//������������������
		for (vector<float>::size_type j = 0; j <= descriptors.size() - 1; ++j)
		{
			featureVectorOfSample.at<float>(i, j) = descriptors[j];
		}

		//�������������
		//!!ע���������һ��Ҫ��int ���͵�
		classOfSample.at<int>(i, 0) = imageClass[i];
	}


	///////////////////////////////////ʹ��SVM������ѵ��///////////////////////////////////////////////////	
	//���ò���
	//�ο�3.0��Demo
	Ptr<ml::SVM> svm = ml::SVM::create();
	svm->setKernel(ml::SVM::RBF);
	svm->setType(ml::SVM::C_SVC);
	svm->setC(10);
	svm->setCoef0(1.0);
	svm->setP(1.0);
	svm->setNu(0.5);
	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON));

	//ʹ��SVMѧϰ         
	svm->train(featureVectorOfSample, ml::ROW_SAMPLE, classOfSample);

	//���������
	svm->save("Classifier.xml");


	///////////////////////////////////ʹ��ѵ���õķ���������ʶ��///////////////////////////////////////////////////
	vector<string> testImagePath;
	ifstream testData(string(FILEPATH) + "TestData.txt", ios::out);
	while (!testData.eof())
	{
		getline(testData, buffer);
		//��ȡ
		if (!buffer.empty())
			testImagePath.push_back(buffer);

	}
	testData.close();

	ofstream fileOfPredictResult(string(FILEPATH) + "PredictResult.txt"); //���ʶ��Ľ��
	for (vector<string>::size_type i = 0; i <= testImagePath.size() - 1; ++i)
	{
		//��ȡ����ͼƬ
		Mat src = imread(testImagePath[i], -1);
		if (src.empty())
		{
			cout << "Can not load the image:" << testImagePath[i] << endl;
			continue;
		}

		//����
		Mat testImage;
		resize(src, testImage, Size(64, 64));

		//����ͼƬ��ȡHOG����
		HOGDescriptor hog(cvSize(64, 64), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
		vector<float> descriptors;
		hog.compute(testImage, descriptors);
		cout << "HOG dimensions:" << descriptors.size() << endl;

		Mat featureVectorOfTestImage(1, descriptors.size(), CV_32FC1);
		for (int j = 0; j <= descriptors.size() - 1; ++j)
		{
			featureVectorOfTestImage.at<float>(0, j) = descriptors[j];
		}

		//�Բ���ͼƬ���з��ಢд���ļ�
		int predictResult = svm->predict(featureVectorOfTestImage);
		char line[512];
		//printf("%s %d\r\n", testImagePath[i].c_str(), predictResult);
		std::sprintf(line, "%s %d\n", testImagePath[i].c_str(), predictResult);
		fileOfPredictResult << line;

	}
	fileOfPredictResult.close();
}

int main()
{
	//HOG+SVMʶ��ʽ1��ֱ��������
	//HOG_SVM1();

	//HOG+SVMʶ��ʽ2�����ͼƬ�еĴ���Ŀ��ľ���
	HOG_SVM2();
}