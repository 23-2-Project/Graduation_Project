#include <chrono>
class Timer {
private:
	std::chrono::steady_clock::time_point start_time;
public:
	Timer() {
		start_time = std::chrono::steady_clock::now();
	}
	float now() const {
		std::chrono::duration<float> dif = std::chrono::steady_clock::now()-start_time;
		return  dif.count();
	}
	void reset() {
		start_time = std::chrono::steady_clock::now();
	}
};