#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <tuple>
#include <fstream>
#include <hip/hip_runtime.h>
#include <sstream>
#include <chrono>
using namespace std;
using namespace std::chrono;

#define index(i,j,m) (((j)*(m))+(i))
int BLOCK_SIZE = 32;





// Getiting copies from  data
__global__ void duplicate(float* dup1, float* dup2, float* input) {

	int i = blockIdx.x;
	int j = threadIdx.x;
	int k = gridDim.x;

	dup1[index(i, j, k)] = input[index(i, j, k)];
	dup2[index(i, j, k)] = input[index(i, j, k)];
}



__global__ void rowSummation(float* result, int m, float* row) {

	int i = threadIdx.x;
	result[index(0, i, 1)] = 0;

	for (int s = 0; s < m; ++s) {
		result[index(0, i, 1)] = row[index(s, i, m)] + result[index(0, i, 1)];
	}
}



__global__ void matmul(float* outMat, int p, float* lhs, float* rhs) {

	int i = blockIdx.x;
	int j = threadIdx.x;
	int k = gridDim.x;

	outMat[index(i, j, k)] = 0;
	for (int s = 0; s < p; ++s) {
		outMat[index(i, j, k)] += lhs[index(s, i, p)] * rhs[index(s, j, p)];
	}
}



__global__ void subtract(float* out, float* in1, float* in2) {

	int i = blockIdx.x;
	int j = threadIdx.x;
	int k = gridDim.x;

	out[index(i, j, k)] = in1[index(i, j, k)] - in2[index(i, j, k)];
}




float L2ErrorSquared(int m, int n, float* Pred, float* True) {

	float a = 0;
	float b = 0;
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			a += pow(Pred[index(i, j, m)] - True[index(i, j, m)], 2);
			b += pow(True[index(i, j, m)], 2);
		}
	}
	float L2Error;
	L2Error = sqrt(a / b);
	return L2Error;
}



// Initializing weight and bias matrix 
void initializer(float* W, float* b, int dim1, int dim2) {

	int t = 0;
	float init;
	t = chrono::system_clock::now().time_since_epoch().count();
	default_random_engine generator(t);
	init = sqrt(2.0 / (dim1 + dim2));
	normal_distribution<float> distribute(0.0, init);

	for (int i = 0; i < dim1; ++i) {
		for (int j = 0; j < dim2; ++j) {
			b[index(0, j, 1)] = 0.0f;
			W[index(i, j, dim1)] = distribute(generator);
		}
	}
}



__global__ void backpropagation(float* actVal, float* BP0, float* W, float* BP, int nL) {

	int i = blockIdx.x;
	int j = threadIdx.x;
	int n = gridDim.x;
	int p = blockDim.x;

	//implementing BP = (1-actVal.actVal).(BP0*W^T) 
	BP[index(i, j, n)] = 0;
	for (int k = 0; k < nL; ++k) {
		BP[index(i, j, n)] = BP0[index(i, k, n)] * W[index(j, k, p)] + BP[index(i, j, n)];
	}

	BP[index(i, j, n)] = (1.0 - actVal[index(i, j, n)] * actVal[index(i, j, n)]) * BP[index(i, j, n)];
}



// Claculating Dense Layer 
__global__ void denseLayer(float* Hidden, float* Activation, int layer, float* X, float* W, float* b) {

	int i = blockIdx.x;
	int j = threadIdx.x;
	int n = gridDim.x;

	// H=X*W+b, (n x q) = (n x p)*(p x q)+(1 x q)
	Hidden[index(i, j, n)] = b[index(0, j, 1)];

	for (int k = 0; k < layer; ++k) {
		Hidden[index(i, j, n)] = Hidden[index(i, j, n)] + X[index(i, k, n)] * W[index(k, j, layer)];
	}

	Activation[index(i, j, n)] = tanh(Hidden[index(i, j, n)]);// A = tanh(H), (n x q) = (n x q)
}



// Adam Optimizer
__global__ void Adam(float* mt, float* vt, float lr, int itr, float* w, float* dW) {

	float eps = 1e-8;
	float beta1 = 0.9;
	float beta2 = 0.999;
	int i = blockIdx.x;
	int j = threadIdx.x;
	int k = gridDim.x;

	vt[index(i, j, k)] = vt[index(i, j, k)] * beta2 + (dW[index(i, j, k)] * dW[index(i, j, k)]) - beta2 * (dW[index(i, j, k)] * dW[index(i, j, k)]);
	float vt_hat = (vt[index(i, j, k)] / (1.0 - pow(beta2, itr)));
	float scal = 1.0 / (sqrt(vt_hat) + eps);
	mt[index(i, j, k)] = mt[index(i, j, k)] * beta1 + dW[index(i, j, k)] - beta1 * dW[index(i, j, k)];
	float mt_hat = (mt[index(i, j, k)] / (1.0 - pow(beta1, itr)));
	w[index(i, j, k)] = w[index(i, j, k)] - scal * mt_hat * lr;

}



// Loading test and train data
tuple< vector<float>, vector<float>, int > load(string name) {

	string data;
	float d1, d2;
	vector<float> input;
	vector<float> target;
	int inSize;
	ifstream file{ name };
	if (!file) {
		cout << "The file does not exist" << name << "\n";
	}
	while (getline(file, data)) {
		istringstream row(data);
		row >> d1 >> d2;
		input.push_back(d1);
		target.push_back(d2);
	}
	inSize = input.size();
	return make_tuple(input, target, inSize);
}



class ANN {

private:

	float* input;
	float* output;
	int xDim;
	int yDim;
	int n; // number of data	
	vector<int> layers; // layers (e.g., {1,10,10,10,1})
	int sizeOfLayers;
	vector<float*> hNode; // hidden noeds
	vector<float*> actNode; // activation nodes
	vector<float*> weights;
	vector<float*> biases;
	vector<float*> bpNode; // backpropagation nodes
	vector<float*> lossWeights; // loss fucntion gradients for weights
	vector<float*> lossBias; // loss function gradients for biases
	// Parameters for Adam optimizer 
	vector<float*> AdWeights1;
	vector<float*> Adbiases1;
	vector<float*> AdWeights2;
	vector<float*> Adbiases2;


public:

	ANN(int _n, float* _input, float* _output, int _xDim, int _yDim, const vector<int>& _layers) : xDim{ _xDim }, yDim{ _yDim }, n{ _n }, layers{ _layers }{

		hipMallocManaged(&input, n* xDim * sizeof(float));
		hipMallocManaged(&output, n* yDim * sizeof(float));
		sizeOfLayers = layers.size();


		//  hidden nodes initialization
		for (int l = 0; l < sizeOfLayers; ++l) {

			float* A;
			hipMallocManaged(&A, n * layers[l] * sizeof(float));
			actNode.push_back(A);
			float* H;
			hipMallocManaged(&H, n * layers[l] * sizeof(float));
			hNode.push_back(H);
		}

		// weights and biases initialization
		for (int l = 0; l < sizeOfLayers - 1; ++l) {

			float* b;
			hipMallocManaged(&b, 1 * layers[l + 1] * sizeof(float));
			float* W;
			hipMallocManaged(&W, layers[l] * layers[l + 1] * sizeof(float));

			initializer(W, b, layers[l], layers[l + 1]);
			biases.push_back(b);
			weights.push_back(W);
		}

		// Adam optimization parameters initialization
		for (int l = 0; l < sizeOfLayers - 1; ++l) {

			float* AdW1;
			hipMallocManaged(&AdW1, layers[l] * layers[l + 1] * sizeof(float));
			AdWeights1.push_back(AdW1);

			float* AdB1;
			hipMallocManaged(&AdB1, 1 * layers[l + 1] * sizeof(float));
			Adbiases1.push_back(AdB1);

			float* AdW2;
			hipMallocManaged(&AdW2, layers[l] * layers[l + 1] * sizeof(float));
			AdWeights2.push_back(AdW2);

			float* AdB2;
			hipMallocManaged(&AdB2, 1 * layers[l + 1] * sizeof(float));
			Adbiases2.push_back(AdB2);
		}

		//  backpropagation initialization
		for (int s = 0; s < sizeOfLayers - 1; ++s) {

			float* D;//derivative
			hipMallocManaged(&D, n * layers[s + 1] * sizeof(float)); // layers[s+1] x n
			bpNode.push_back(D);

			float* wLoss;
			hipMallocManaged(&wLoss, layers[s] * layers[s + 1] * sizeof(float)); // layers[s+1] x layers[l]
			lossWeights.push_back(wLoss);

			float* bLoss;
			hipMallocManaged(&bLoss, 1 * layers[s + 1] * sizeof(float)); // layers[s+1] x 1
			lossBias.push_back(bLoss);
		}

		//copying data
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < xDim; ++j)
				input[index(i, j, n)] = _input[index(i, j, n)];
			for (int k = 0; k < yDim; ++k)
				output[index(i, k, n)] = _output[index(i, k, n)];
		}
	}



	void train(int numberOfIterations, float lr) {
		chrono::duration<float> dur;
		chrono::duration<float> durTotal;
		auto t1 = high_resolution_clock::now();
		auto t2 = high_resolution_clock::now();
		auto t3 = high_resolution_clock::now();
		auto t4 = high_resolution_clock::now();
		ofstream file;
		file.open("LossVals.txt");
		if (!file) {
			cerr << "can't open output file" << endl;
		}

		for (int it = 1; it < numberOfIterations + 1; ++it) {

			lossFunction();//getting loss values and derivatives
			for (int l = 0; l < sizeOfLayers - 1; ++l) {
				// Updating
				Adam << <layers[l], layers[l + 1] >> > (AdWeights2[l], AdWeights1[l], lr, it, weights[l], lossWeights[l]);
				Adam << <1, layers[l + 1] >> > (Adbiases2[l], Adbiases1[l], lr, it, biases[l], lossBias[l]);
			}

			hipDeviceSynchronize();
			float loss = 0.5 * L2ErrorSquared(n, yDim, hNode[sizeOfLayers - 1], output);

			if (it % 10 == 0) {

				float loss = 0.5 * L2ErrorSquared(n, yDim, hNode[sizeOfLayers - 1], output);
				t2 = high_resolution_clock::now();
				dur = t2 - t1;
				cout << "iteration: " << it
					<< ", loss: " << loss
					<< ", time: " << dur.count()
					<< ", learning rate: " << lr << "\n";
				t1 = high_resolution_clock::now();

			}
			file << loss << "\n"; //writing loss values to a file
		}

		t4 = high_resolution_clock::now();
		durTotal = t4 - t3;
		cout << "The totla time for this model was: " << durTotal.count() << endl;
		t3 = high_resolution_clock::now();
		file.close();
	}



	void lossFunction() {

		// Feed forward and Backpropagation steps 
		duplicate << <n, xDim >> > (hNode[0], actNode[0], input);
		for (int layernumber = 0; layernumber < sizeOfLayers - 1; layernumber++) {
			denseLayer << <n, layers[layernumber + 1] >> > (hNode[layernumber + 1], actNode[layernumber + 1], layers[layernumber], actNode[layernumber], weights[layernumber], biases[layernumber]);
		}
		subtract << <n, layers[sizeOfLayers - 1] >> > (bpNode[sizeOfLayers - 2], hNode[sizeOfLayers - 1], output);

		for (int layernumber = sizeOfLayers - 2; layernumber > 0; layernumber--) {

			matmul << <layers[layernumber], layers[layernumber + 1] >> > (lossWeights[layernumber], n, hNode[layernumber], bpNode[layernumber]);
			rowSummation << <1, layers[layernumber + 1] >> > (lossBias[layernumber], n, bpNode[layernumber]);
			backpropagation << <n, layers[layernumber] >> > (actNode[layernumber], bpNode[layernumber], weights[layernumber], bpNode[layernumber - 1], layers[layernumber + 1]);
		}
		matmul << <layers[0], layers[1] >> > (lossWeights[0], n, hNode[0], bpNode[0]);
		rowSummation << <1, layers[1] >> > (lossBias[0], n, bpNode[0]);
	}



	void test(int numDataTest, float* xTest, float* yPred) {

		vector<float*> hidVals;
		vector<float*> actVals;
		for (int l = 0; l < sizeOfLayers; ++l) {

			float* hNode;
			hipMallocManaged(&hNode, numDataTest * layers[l] * sizeof(float));
			hidVals.push_back(hNode);

			float* aNode;
			hipMallocManaged(&aNode, numDataTest * layers[l] * sizeof(float));
			actVals.push_back(aNode);
		}
		duplicate << <numDataTest, xDim >> > (hidVals[0], actVals[0], xTest);

		for (int l = 0; l < sizeOfLayers - 1; ++l) {
			denseLayer << <numDataTest, layers[l + 1] >> > (hidVals[l + 1], actVals[l + 1], layers[l], actVals[l], weights[l], biases[l]);
		}

		hipDeviceSynchronize();

		for (int i = 0; i < numDataTest; ++i) {
			for (int k = 0; k < yDim; ++k) {
				yPred[index(i, k, n)] = (hidVals[sizeOfLayers - 1])[index(i, k, n)];
			}
		}

		for (int s = 0; s < sizeOfLayers; ++s) {
			hipFree(hidVals[s]);
			hipFree(actVals[s]);
		}
	}
};




int main() {

	// Loding data for training and testing of the network
	auto trainData = load("./TrainData.csv");
	auto testData = load("./TestData.csv");

	int NumOfTrainData = get<2>(trainData);
	int NumOfTestData = get<2>(testData);

	vector<float> xVecTrainData = get<0>(trainData);
	vector<float> yVecTrainData = get<1>(trainData);
	vector<float> xVecTestdata = get<0>(testData);
	vector<float> yVecTestdata = get<1>(testData);

	float* xTrain; hipMallocManaged(&xTrain, NumOfTrainData * 1 * sizeof(float));
	float* yTrain; hipMallocManaged(&yTrain, NumOfTrainData * 1 * sizeof(float));
	float* xTest; hipMallocManaged(&xTest, NumOfTestData * 1 * sizeof(float));
	float* yTest; hipMallocManaged(&yTest, NumOfTestData * 1 * sizeof(float));

	for (int i = 0; i < NumOfTrainData; ++i) {
		xTrain[i] = xVecTrainData[i];
		yTrain[i] = yVecTrainData[i];
	}

	for (int i = 0; i < NumOfTestData; ++i) {
		xTest[i] = xVecTestdata[i];
		yTest[i] = yVecTestdata[i];
	}

	// Making the ANN model
	ANN model(NumOfTrainData, xTrain, yTrain, 1, 1, { 1,10,10,1 });

	// Training 
	model.train(10000, 1e-3); //numberOfIterations, learning rate

	// Testing 
	float* yPred; hipMallocManaged(&yPred, NumOfTestData * 1 * sizeof(float));
	model.test(NumOfTestData, xTest, yPred);

	//Printing L2 error
	cout << "The L2 error is :" << L2ErrorSquared(NumOfTestData, 1, yPred, yTest) << "\n";

	// Saving wave height preditions
	ofstream fileResults;
	fileResults.open("results.txt");
	if (!fileResults) {
		cerr << "can't open output file" << endl;
	}

	for (int i = 0; i < NumOfTestData; ++i) {

		fileResults << xTest[i] << " " << yTest[i] << " " << yPred[i] << "\n";
	}

	fileResults.close();

	return 0;
}
