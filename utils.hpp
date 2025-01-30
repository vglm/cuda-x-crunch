#pragma once

#include <string>
#include <cstdint>

std::string string_replace(const std::string & str, const std::string & from, const std::string & to);
std::string string_to_lower(const std::string & str);

std::string normalize_ethereum_address(const std::string & address);

// Convert 20 bytes array to ethereum address like 0x1234567890123456789012345678901234567890
std::string bytes_to_ethereum_address(const uint8_t *bytes);

// Get application time with nanosecond precision
double get_app_time_sec();

std::string get_utc_time();