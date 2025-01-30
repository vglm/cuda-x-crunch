#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <fstream>
#include <string>
#include <mutex>
#include <cstdarg>
#include <cstdio>

class Logger {
public:
    enum Level {
        ERROR = 0,
        WARNING = 1,
        INFO = 2,
        DEBUG = 3
    };

    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    void setLevel(Level level) {
        logLevel = level;
    }

    void setLogFile(const std::string& filename) {
        std::lock_guard<std::mutex> lock(logMutex);
        logFile.open(filename, std::ios::out | std::ios::app);
    }

    void log(Level level, const char* format, ...) {
        if (level <= logLevel) {
            std::lock_guard<std::mutex> lock(logMutex);
            char buffer[1024];
            va_list args;
            va_start(args, format);
            vsnprintf(buffer, sizeof(buffer), format, args);
            va_end(args);

            std::ostream& out = logFile.is_open() ? logFile : std::cerr;
            out << "[" << get_utc_time() << getLabel(level) << buffer << std::endl;
        }
    }

private:
    Level logLevel = INFO;
    std::ofstream logFile;
    std::mutex logMutex;

    Logger() = default;
    ~Logger() { if (logFile.is_open()) logFile.close(); }
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    std::string getLabel(Level level) {
        switch (level) {
            case ERROR: return " ERROR] ";
            case WARNING: return " WARNING] ";
            case INFO: return " INFO] ";
            case DEBUG: return " DEBUG] ";
            default: return " LOG] ";
        }
    }
};

#define LOG_ERROR(...) Logger::getInstance().log(Logger::ERROR, __VA_ARGS__)
#define LOG_WARNING(...) Logger::getInstance().log(Logger::WARNING, __VA_ARGS__)
#define LOG_INFO(...) Logger::getInstance().log(Logger::INFO, __VA_ARGS__)
#define LOG_DEBUG(...) Logger::getInstance().log(Logger::DEBUG, __VA_ARGS__)

#define CHECK_CUDA_ERROR(comment) { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        LOG_ERROR("CUDA FAILED: %s in %s:%d: %s", comment, __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

#endif // LOGGER_H
