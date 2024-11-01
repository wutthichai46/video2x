#ifndef REALSRGAN_FILTER_H
#define REALSRGAN_FILTER_H

#include <filesystem>

extern "C" {
#include <libavcodec/avcodec.h>
}

#include "filter.h"
#include "realesrgan.h"

// RealesrganFilter class definition
class RealesrganFilter : public Filter {
   private:
    RealESRGAN *realesrgan;
    int gpuid;
    bool tta_mode;
    int scaling_factor;
    const char *model;
    const std::filesystem::path custom_model_param_path;
    const std::filesystem::path custom_model_bin_path;
    AVRational in_time_base;
    AVRational out_time_base;
    AVPixelFormat out_pix_fmt;

   public:
    // Constructor
    RealesrganFilter(
        int gpuid = 0,
        bool tta_mode = false,
        int scaling_factor = 4,
        const char *model = "realesr-animevideov3",
        const std::filesystem::path custom_model_param_path = std::filesystem::path(),
        const std::filesystem::path custom_model_bin_path = std::filesystem::path()
    );

    // Destructor
    virtual ~RealesrganFilter() override;

    // Initializes the filter with decoder and encoder contexts
    int init(AVCodecContext *dec_ctx, AVCodecContext *enc_ctx, AVBufferRef *hw_ctx) override;

    // Processes an input frame and returns the processed frame
    int process_frame(AVFrame *in_frame, AVFrame **out_frame) override;
};

#endif
