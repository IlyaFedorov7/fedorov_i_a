## Работа 1. Исследование гамма-коррекции
автор: Федоров Илья Андреевич БПМ-19-1
дата: 2022-02-17T21:02:54

<!-- url: https://gitlab.com/2021-misis-spring/polevoy_d_v/-/tree/master/prj.labs/lab01 -->

### Задание
1. Сгенерировать серое тестовое изображение $I_1$ в виде прямоугольника размером 768х60 пикселя с плавным изменение пикселей от черного к белому, одна градация серого занимает 3 пикселя по горизонтали.
2. Применить  к изображению $I_1$ гамма-коррекцию с коэффициентом из интервала 2.2-2.4 и получить изображение $G_1$ при помощи функци pow.
3. Применить  к изображению $I_1$ гамма-коррекцию с коэффициентом из интервала 2.2-2.4 и получить изображение $G_2$ при помощи прямого обращения к пикселям.
4. Показать визуализацию результатов в виде одного изображения (сверху вниз $I_1$, $G_1$, $G_2$).
5. Сделать замер времени обработки изображений в п.2 и п.3, результаты отфиксировать в отчете.

### Результаты

![](lab01.png)
Рис. 1. Результаты работы программы (сверху вниз $I_1$, $G_1$, $G_2$)

### Текст программы

```cpp
#include <opencv2/opencv.hpp>

using namespace std::chrono;

int main() {
	cv::Mat img(180, 768, CV_8UC1);
	
	img = 0;
	cv::Rect2d rc = { 0, 0, 768, 60 };
	double alpha = 2.2;
	double beta = 2.4;
	//cv::rectangle(img, rc, { 100 }, 1);
	
	for (int y = 0; y < 180; y++)
	{
		for (int x = 0; x < 768; x++)
		{
			img.at<uchar>(y, x) = x / 3;
		}
	}
	
	rc.y += rc.height;
	auto start = high_resolution_clock::now();
	cv::Mat new_img{ img };
	new_img.convertTo(new_img, CV_64FC1, 1.0f / 255.0f);
	cv::pow(new_img, alpha, new_img);
	new_img.convertTo(new_img, CV_64FC1, 255.0f);
	new_img(rc).copyTo(img(rc));
	auto finish = high_resolution_clock::now();
	auto elapsed = duration_cast<microseconds>(finish - start);
	std::cout � elapsed.count() � "microseconds" � std::endl;

	//3
	//cv::rectangle(img, rc, { 100 }, 1);
	rc.y += rc.height;
	auto start1 = high_resolution_clock::now();
	for (int y = 120; y < 180; y++) {
		for (int x = 0; x < 768; x++) {
			img.at<cv::uint8_t>(y, x) = cv::saturate_cast<uchar>(cv::pow(img.at<cv::uint8_t>(y, x) / 255.0f, beta) * 255.0f);

		}
	}
	auto finish1 = high_resolution_clock::now();
	auto elapsed1 = duration_cast<microseconds>(finish1 - start1);
	std::cout � elapsed1.count() � "microseconds" � std::endl;
	//cv::rectangle(img, rc, { 100 }, 1);

	// save result
	//cv::imwrite("lab0.png", img);

	return 0;
}
