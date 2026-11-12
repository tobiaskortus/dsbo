#pragma once

#include <atomic>
#include <chrono>
#include <filesystem>
#include <random>
#include <string>

namespace dsbo::test {

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
