#ifndef LIBVIDEO2X_H
#define LIBVIDEO2X_H

#include <stdbool.h>
#include <stdint.h>
#include <time.h>

#ifdef _WIN32
#ifdef LIBVIDEO2X_EXPORTS
#define LIBVIDEO2X_API __declspec(dllexport)
#else
#define LIBVIDEO2X_API __declspec(dllimport)
#endif
#else
#define LIBVIDEO2X_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>

// Enum to specify filter type
enum FilterBackend {
    FILTER_BACKEND_LIBPLACEBO,
    FILTER_BACKEND_REALESRGAN,
    FILTER_BACKEND_RIFE
};

// Enum to specify log level
enum Libvideo2xLogLevel {
    LIBVIDEO2X_LOG_LEVEL_TRACE,
    LIBVIDEO2X_LOG_LEVEL_DEBUG,
    LIBVIDEO2X_LOG_LEVEL_INFO,
    LIBVIDEO2X_LOG_LEVEL_WARNING,
    LIBVIDEO2X_LOG_LEVEL_ERROR,
    LIBVIDEO2X_LOG_LEVEL_CRITICAL,
    LIBVIDEO2X_LOG_LEVEL_OFF
};

// Configuration for Libplacebo filter
struct LibplaceboConfig {
    int output_width;
    int output_height;
    const char *shader_path;
};

// Configuration for RealESRGAN filter
struct RealESRGANConfig {
    int gpuid;
    bool tta_mode;
    int scaling_factor;
    const char *model;
};

// Configuration for RIFE filter
struct RIFEConfig {
    int gpuid;
    bool tta_mode;
    bool tta_temporal_mode;
    bool uhd_mode;
    int num_threads;
    bool rife_v2;
    bool rife_v4;
    const char *model_dir;
};

// Unified filter configuration
struct FilterConfig {
    enum FilterBackend filter_backend;
    union {
        struct LibplaceboConfig libplacebo;
        struct RealESRGANConfig realesrgan;
        struct RIFEConfig rife;
    } config;
};

// Encoder configuration
struct EncoderConfig {
    int output_width;
    int output_height;
    struct AVRational frame_rate;
    bool copy_streams;
    enum AVCodecID codec;
    enum AVPixelFormat pix_fmt;
    const char *preset;
    int64_t bit_rate;
    float crf;
};

// Video processing context
struct VideoProcessingContext {
    int64_t processed_frames;
    int64_t total_frames;
    time_t start_time;
    bool pause;
    bool abort;
    bool completed;
};

// C-compatible process_video function
LIBVIDEO2X_API int process_video(
    const char *input_filename,
    const char *output_filename,
    enum Libvideo2xLogLevel log_level,
    bool benchmark,
    enum AVHWDeviceType hw_device_type,
    const struct FilterConfig *filter_config,
    struct EncoderConfig *encoder_config,
    struct VideoProcessingContext *proc_ctx
);

#ifdef __cplusplus
}
#endif

#endif  // LIBVIDEO2X_H
