#include <opencv2/opencv.hpp>

using namespace std::chrono;

int main() {
	cv::Mat image(180, 768, CV_8UC1);

	image = 0;
	cv::Rect2d rc = { 0, 0, 768, 60 };
	double a = 2.2;
	double b = 2.4;
	

	for (int y = 0; y < 180; y++)
	{
		for (int x = 0; x < 768; x++)
		{
			image.at<uchar>(y, x) = x / 3;
		}
	}

	rc.y += rc.height;
	auto start = high_resolution_clock::now();
	cv::Mat new_image{ image };
	new_image.convertTo(new_image, CV_64FC1, 1.0f / 255.0f);
	cv::pow(new_image, a, new_image);
	new_image.convertTo(new_image, CV_64FC1, 255.0f);
	new_image(rc).copyTo(image(rc));
	auto finish = high_resolution_clock::now();
	auto elapsed = duration_cast<microseconds>(finish - start);
	std::cout << elapsed.count() << " microseconds" << std::endl;

	
	rc.y += rc.height;
	auto start1 = high_resolution_clock::now();
	for (int y = 120; y < 180; y++) {
		for (int x = 0; x < 768; x++) {
			image.at<cv::uint8_t>(y, x) = cv::saturate_cast<uchar>(cv::pow(image.at<cv::uint8_t>(y, x) / 255.0f, b) * 255.0f);

		}
	}
	auto finish1 = high_resolution_clock::now();
	auto elapsed1 = duration_cast<microseconds>(finish1 - start1);
	std::cout << elapsed1.count() << " microseconds" << std::endl;
	
	cv::imwrite("lab0.png", image);

	return 0;
}