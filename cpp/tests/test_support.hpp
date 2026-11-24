#pragma once

#include <dsbo/hdf5/types.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace dsbo::test {

inline bool same_bytes(const io::Array& left, const io::Array& right) {
    return left.nbytes() == right.nbytes() &&
           (!left.nbytes() || std::memcmp(left.data(), right.data(), left.nbytes()) == 0);
}

template<class T> bool same_values(const io::Array& actual, const std::vector<T>& expected) {
    if (actual.type != io::dtype<T>() || actual.size() != expected.size()) return false;
    if (expected.empty()) return true;
    if constexpr (std::is_same_v<T, bool>)
        return std::equal(expected.begin(), expected.end(), actual.data(),
                          [](bool value, unsigned char byte) { return value == (byte != 0); });
    else return std::equal(expected.begin(), expected.end(), actual.data<T>());
}

class TestDirectory {
public:
    TestDirectory() {
        static std::atomic<unsigned long long> sequence{0};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        std::random_device random;
        root_ = std::filesystem::temp_directory_path() /
                ("dsbo-native-test-" + std::to_string(stamp) + "-" +
                 std::to_string(random()) + "-" + std::to_string(sequence++));
        std::error_code error;
        std::filesystem::remove_all(root_, error);
        std::filesystem::create_directories(root_);
    }

    ~TestDirectory() {
        std::error_code error;
        std::filesystem::remove_all(root_, error);
    }

    TestDirectory(const TestDirectory&) = delete;
    TestDirectory& operator=(const TestDirectory&) = delete;

    std::string operator()(const std::string& name) const {
        return (root_ / name).string();
    }

    const std::filesystem::path& path() const {
        return root_;
    }

private:
    std::filesystem::path root_;
};

} // namespace dsbo::test
