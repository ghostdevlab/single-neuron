#include <iostream>

class Perceptron {
public:
    Perceptron(int _n, const float *_w, float (*_activation)(float)):
        n(_n), w(std::make_unique<float[]>(_n)), activation(_activation) {
        std::copy(_w, _w + _n, w.get());
    }

    float calculate(const float* input) const;
private:
    std::unique_ptr<float[]> w;
    int n;
    float (*activation)(float);
};

float Perceptron::calculate(const float *input) const {
    float sum = 0.0f;

    for(int i = 0; i<n; i++) sum += w[i] * input[i];

    return activation(sum);
}

float sigmoidActivation(float x) {
    return 1.0f / (1 + exp(-x));
}

int main() {
    float w[] = { 5, 3, 2 };
    float input[] = { -2, 4, 1 };

    Perceptron p(3, w, sigmoidActivation);

    std::cout<<"Value: "<<p.calculate(input)<<std::endl;

    return 0;
}
