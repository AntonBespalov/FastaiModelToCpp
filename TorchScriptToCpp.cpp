#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

int main(int argc, const char* argv[]) {

	torch::jit::script::Module module;
	try {
		// Deserialize the ScriptModule from a file using torch::jit::load().
		module = torch::jit::load(argv[1]);
	}
	catch (const c10::Error & e) {
		std::cerr << "error loading the model\n";
		return -1;
	}
	   
	// Read the image file
	Mat image = imread(argv[2], IMREAD_UNCHANGED);

	// Check for failure
	if (image.empty())
	{
		cout << "Could not open or find the image" << endl;
		cin.get(); //wait for any key press
		return -1;
	}

	cv::cvtColor(image, image, COLOR_BGR2RGB);

	// Convert image to tensor form
	image.convertTo(image, CV_32FC3, 1.0f / 255.0f);
	auto input_tensor = torch::from_blob(image.data, { 1, image.rows, image.cols, /*channels*/3 });
	input_tensor = input_tensor.permute({ 0, 3, 1, 2 });

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(input_tensor);

	// Forward pass
	at::Tensor out = module.forward(inputs).toTensor();
	at::Tensor out_softmax = torch::softmax(out, /*dim=*/-1);
	std::cout << out_softmax << '\n';
}
