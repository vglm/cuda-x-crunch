#include "utils.hpp"
#include <iostream>
#include <string>
#include <algorithm>
#include <cctype>
#include <chrono>

std::string string_replace(const std::string & str, const std::string & from, const std::string & to)
{
    std::string ret = str;
    size_t start_pos = 0;
    while((start_pos = ret.find(from, start_pos)) != std::string::npos) {
        ret.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
        return ret;
    }

std::string string_to_lower(const std::string & str)
{
    std::string ret = str;
    std::transform(ret.begin(), ret.end(), ret.begin(), [](unsigned char c){ return std::tolower(c); });
    return ret;
}

std::string normalize_ethereum_address(const std::string & address)
{
    std::string ret = address;
    ret = string_replace(ret, "0x", "");
    ret = string_to_lower(ret);
    //check if ret contains only hex characters
    if (ret.find_first_not_of("0123456789abcdef") != std::string::npos) {
        std::cerr << "Invalid ethereum address, only hex characters are allowed: " << address << std::endl;
        return "";
    }
    if (ret.length() != 40) {
        std::cerr << "Invalid ethereum address, length has to be 40, given length " << ret.length() << std::endl;
        return "";
    }
    return ret;
}

const uint64_t start = std::chrono::nanoseconds(std::chrono::high_resolution_clock::now().time_since_epoch()).count();;
double get_current_timestamp() {

    auto now = std::chrono::nanoseconds(std::chrono::high_resolution_clock::now().time_since_epoch()).count();;
    return (now - start) / 1.0E9;
}

std::string bytes_to_ethereum_address(const uint8_t *bytes) {
    char buf[43];
    buf[0] = '0';
    buf[1] = 'x';
    for (int i = 0; i < 20; i++) {
        sprintf(buf + i * 2 + 2, "%02x", bytes[i]);
    }
    return std::string(buf, 42);
}
