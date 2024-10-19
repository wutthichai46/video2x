#ifndef FILTER_H
#define FILTER_H

#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavutil/buffer.h>
}

enum FilterOperation {
    FILTER_OPERATION_UPSCALING,
    FILTER_OPERATION_INTERPOLATION,
};

// Abstract base class for filters
class Filter {
   public:
    virtual ~Filter() = default;

    virtual FilterOperation get_type() const = 0;

    virtual int init(AVCodecContext *dec_ctx, AVCodecContext *enc_ctx, AVBufferRef *hw_ctx) = 0;

    virtual int flush(std::vector<AVFrame *> &processed_frames) { return 0; }
};

class UpscalingFilter : public Filter {
   public:
    FilterOperation get_type() const override { return FILTER_OPERATION_UPSCALING; }

    virtual int upscale(AVFrame *prev_frame, AVFrame *in_frame, AVFrame **out_frame) = 0;
};

class InterpolationFilter : public Filter {
   public:
    FilterOperation get_type() const override { return FILTER_OPERATION_INTERPOLATION; }

    virtual int
    interpolate(AVFrame *prev_frame, AVFrame *in_frame, float time_step, AVFrame **out_frame) = 0;
};

#endif  // FILTER_H
