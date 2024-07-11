#include <iostream>
#include <SDL.h>

typedef struct {
    float (*activation)(float);
    float (*derivation)(float);
} ActivationFunction;

class Perceptron {
public:
    Perceptron(int _n, ActivationFunction& activationFunction);

    float calculate(const float* input) const;

    void training(const float* input, float expected, float step);

    float inputSum(const float* input) const;

private:
    int n;
    std::unique_ptr<float[]> w;
    float (*activation)(float);
    float (*derivation)(float);
};

class Window {
public:
    Window(const char *name, int width, int height);

    void clear(Uint32 color);

    void lock();
    void unlock();

    void updateWindow();

    void putPixel(int x, int y, Uint32 color);
    void putPixel(int x, int y, int size, Uint32 color);

    int getWidth() const;
    int getHeight() const;

    ~Window();
private:
    int w, h;
    SDL_Window *window;
    SDL_Surface *screen;
    SDL_Surface *pixels;
};

float linearActivation(float x);
float linearDerivation(float x);

float sigmoidActivation(float x);
float sigmoidDerivation(float x);

ActivationFunction functions[] = {
        { linearActivation, linearDerivation },
        { sigmoidActivation, sigmoidDerivation}
};

float* inputSet(int count, int testFunction(float x, float y)) {
    auto data = new float [count * 3];

    for(int i=0; i<count; i++) {
        data[i * 3 + 0] = ((float)rand()/ RAND_MAX);
        data[i * 3 + 1] = ((float)rand()/ RAND_MAX);
        data[i * 3 + 2] = testFunction(data[i * 3 + 0], data[i * 3 + 1]);
    }

    return data;
}

int testF(float x, float y) {
    float fy = 0.7f - 0.55f * x;
    return y > fy ? 1 : 0;
}

int main() {
    SDL_Init(SDL_INIT_VIDEO);

    bool running = true;

    Window window("Neuron test", 800, 600);
    auto testSet = std::unique_ptr<float[]>(inputSet(60, testF));
    auto trainingSet = std::unique_ptr<float[]>(inputSet(1000, testF));

    Perceptron perceptron(2, functions[1]);

    int iterations = 0;

    while (running) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) {
                running = false;
            }
        }

        window.clear(0);
        window.lock();

        for(int y = 0; y<window.getHeight(); y++) {
            for(int x=0; x<window.getWidth(); x++) {
                float input[] = { (float)x / window.getWidth(), (float)y / window.getHeight() };
                int color = perceptron.calculate(input) > 0.5 ? (63 << 5) : 31;

                window.putPixel(x, window.getHeight() - 1 - y, 2, color);
            }
        }

        for(int i = 0; i<60; i++) {
            float x = testSet[i * 3 + 0];
            float y = 1.0f - testSet[i * 3 + 1];

            int sx = x * (window.getWidth() - 1);
            int sy = y * (window.getHeight() - 1);

            int color = testSet[i * 3 + 2] > 0.5 ? 0xFFFF : 0x0000;

            window.putPixel(sx, sy, 2, color);
        }

        float step = 0.01f;
        if (iterations < 50) {
            step = 5.0f;
        } else if (iterations < 500) {
            step = 1.0f;
        } else if (iterations < 1000) {
            step = 0.1f;
        } else {
            step = 100.0f/iterations;
        }

        for(int i = 0; i<1000; i++) {
            float x = trainingSet[i * 3 + 0];
            float y = trainingSet[i * 3 + 1];
            float expected = trainingSet[i * 3 + 2];

            float input[] = { x, y };

            perceptron.training(input, expected, step);
        }

        iterations++;

        window.unlock();
        window.updateWindow();
    }

    SDL_Quit();
    return 0;
}

Perceptron::Perceptron(int _n, ActivationFunction &activationFunction) :
    n(_n), w(std::make_unique<float[]>(_n)),
    activation(activationFunction.activation), derivation(activationFunction.derivation) {

    for(int i = 0; i<=n; i++) {
        w[i] = 2.0f * ((float)rand()/RAND_MAX) - 1.0f;
    }

}

float Perceptron::calculate(const float* input) const {
    return activation(inputSum(input));
}

void Perceptron::training(const float* input, float expected, float step) {
    float sum = inputSum(input);
    float result = activation(sum);

    float error = expected - result;
    float de = derivation(sum);

    for(int i = 0; i<n; i++) {
        w[i] += step * error * de * input[i];
    }

    w[n] += step * error * de * 1.0f;
}

float Perceptron::inputSum(const float *input) const {
    float sum = 0.0f;

    for(int i = 0; i<n; i++) {
        sum += input[i] * w[i];
    }

    sum += 1.0f * w[n];

    return sum;
}


Window::Window(const char *name, int width, int height) : w(width), h(height) {
    window = SDL_CreateWindow(
            name,
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            width, height,
            SDL_WINDOW_SHOWN);

    screen = SDL_GetWindowSurface(window);
    pixels = SDL_CreateRGBSurfaceWithFormat(0, width, height, 16, SDL_PIXELFORMAT_RGB565);
}

Window::~Window() {
    SDL_DestroyWindow(window);
}

void Window::clear(Uint32 color) {
    SDL_FillRect(pixels, NULL, color);
}

void Window::lock() {
    SDL_LockSurface(pixels);
}

void Window::unlock() {
    SDL_UnlockSurface(pixels);
}

int Window::getWidth() const { return w; }
int Window::getHeight() const { return h; }

void Window::putPixel(int x, int y, Uint32 color) {
    ((unsigned short*)(((unsigned char*)pixels->pixels) + y * pixels->pitch))[x] = color;
}

void Window::putPixel(int x, int y, int size, Uint32 color) {
    for(int dy = y - size; dy < y + size; dy ++) {
        for (int dx = x - size; dx < x + size; dx++) {
            ((unsigned short *) (((unsigned char *) pixels->pixels) +
                                 (dy) * pixels->pitch))[dx] = color;
        }
    }
}

void Window::updateWindow() {
    SDL_BlitSurface(pixels, NULL, screen, NULL);
    SDL_UpdateWindowSurface(window);
}

float linearActivation(float x) { return x; }
float linearDerivation(float x) { return 1; }

float sigmoidActivation(float x) { return  1.0f/(1.0f + exp(-x)); }
float sigmoidDerivation(float x) { return sigmoidActivation(x) * (1.0f - sigmoidActivation(x)); }