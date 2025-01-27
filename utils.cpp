#include "utils.hpp"


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
