#ifndef RIFE_FILTER_H
#define RIFE_FILTER_H

#include <filesystem>

extern "C" {
#include <libavcodec/avcodec.h>
}

#include "filter.h"
#include "rife.h"

// RealesrganFilter class definition
class RifeFilter : public InterpolationFilter {
   private:
    RIFE *rife;
    int gpuid;
    bool tta_mode;
    bool tta_temporal_mode;
    bool uhd_mode;
    int num_threads;
    bool rife_v2 = false;
    bool rife_v4 = false;
    const std::filesystem::path model_dir_path;
    AVRational input_time_base;
    AVRational output_time_base;
    AVPixelFormat output_pix_fmt;

   public:
    // Constructor
    RifeFilter(
        int gpuid,
        bool tta_mode,
        bool tta_temporal_mode,
        bool uhd_mode,
        int num_threads,
        bool rife_v2,
        bool rife_v4,
        const std::filesystem::path &model_dir_path
    );

    // Destructor
    virtual ~RifeFilter();

    // Defines the filter type
    FilterOperation get_type() const override { return FILTER_OPERATION_INTERPOLATION; }

    // Initializes the filter with decoder and encoder contexts
    int init(AVCodecContext *dec_ctx, AVCodecContext *enc_ctx, AVBufferRef *hw_ctx) override;

    // Process two input frames and interpolate between them
    int interpolate(AVFrame *prev_frame, AVFrame *in_frame, float time_step, AVFrame **out_frame)
        override;
};

#endif  // RIFE_FILTER_H
