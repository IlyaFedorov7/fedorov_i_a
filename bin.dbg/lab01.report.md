## Работа 1. Исследование гамма-коррекции
автор: Федоров Илья Андреевич БПМ-19-1
дата: 2022-02-17

<https://github.com/IlyaFedorov7/fedorov_i_a/tree/master/prj.labs/lab01>

### Задание
1. Сгенерировать серое тестовое изображение $I_1$ в виде прямоугольника размером 768х60 пикселя с плавным изменение пикселей от черного к белому, одна градация серого занимает 3 пикселя по горизонтали.
2. Применить  к изображению $I_1$ гамма-коррекцию с коэффициентом из интервала 2.2-2.4 и получить изображение $G_1$ при помощи функции pow.
3. Применить  к изображению $I_1$ гамма-коррекцию с коэффициентом из интервала 2.2-2.4 и получить изображение $G_2$ при помощи прямого обращения к пикселям.
4. Показать визуализацию результатов в виде одного изображения (сверху вниз $I_1$, $G_1$, $G_2$).
5. Сделать замер времени обработки изображений в п.2 и п.3, результаты зафиксировать в отчете.

### Результаты

![](C:\fedorov_i_a\bin.dbg\lab0.png)
Рис. 1. Результаты работы программы (сверху вниз $I_1$, $G_1$, $G_2$)
### Замеры времени
В первом случае: 9697 microseconds
Во втором случае: 6069 microseconds
### Текст программы

```cpp

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
