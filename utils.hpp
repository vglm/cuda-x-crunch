#pragma once

#include <string>


std::string string_replace(const std::string & str, const std::string & from, const std::string & to);
std::string string_to_lower(const std::string & str);

std::string normalize_ethereum_address(const std::string & address);

double get_current_timestamp();
