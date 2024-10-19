#include "rife_filter.h"

#include <cstdio>
#include <filesystem>

#include <spdlog/spdlog.h>

#include "conversions.h"
#include "fsutils.h"

RifeFilter::RifeFilter(
    int gpuid,
    bool tta_mode,
    bool tta_temporal_mode,
    bool uhd_mode,
    int num_threads,
    bool rife_v2,
    bool rife_v4,
    const std::filesystem::path &model_dir_path
)
    : rife(nullptr),
      gpuid(gpuid),
      tta_mode(tta_mode),
      tta_temporal_mode(tta_temporal_mode),
      uhd_mode(uhd_mode),
      num_threads(num_threads),
      rife_v2(rife_v2),
      rife_v4(rife_v4),
      model_dir_path(std::move(model_dir_path)) {}

RifeFilter::~RifeFilter() {
    if (rife) {
        delete rife;
        rife = nullptr;
    }
}

int RifeFilter::init(AVCodecContext *dec_ctx, AVCodecContext *enc_ctx, AVBufferRef *hw_ctx) {
    // Construct the model directory path using std::filesystem
    std::filesystem::path model_dir_full_path;
    if (std::filesystem::exists(model_dir_path) && std::filesystem::is_directory(model_dir_path)) {
        // If the model directory path is directly readable, use it
        model_dir_full_path = model_dir_path;
    } else {
        // Construct the fallback path using std::filesystem
        model_dir_full_path = find_resource_file(std::filesystem::path("models") / model_dir_path);
    }

    // Check if the model files exist
    if (!std::filesystem::exists(model_dir_full_path)) {
        spdlog::error("RIFE model param file not found: {}", model_dir_full_path.string());
        return -1;
    }

    // Create a new RIFE instance
    // TODO: REMOVE
    rife_v4 = true;
    rife = new RIFE(gpuid, tta_mode, tta_temporal_mode, uhd_mode, num_threads, rife_v2, rife_v4);

    // Store the time bases
    input_time_base = dec_ctx->time_base;
    output_time_base = enc_ctx->time_base;
    output_pix_fmt = enc_ctx->pix_fmt;

    // Load the model
    if (rife->load(model_dir_full_path) != 0) {
        spdlog::error("Failed to load RIFE model");
        return -1;
    }

    return 0;
}

int RifeFilter::interpolate(
    AVFrame *prev_frame,
    AVFrame *in_frame,
    float timestep,
    AVFrame **out_frame
) {
    int ret;

    // Convert the first input frame to ncnn::Mat
    ncnn::Mat frame0_mat = avframe_to_ncnn_mat(prev_frame);
    if (frame0_mat.empty()) {
        spdlog::error("Failed to convert AVFrame to ncnn::Mat");
        return -1;
    }

    // Convert the second input frame to ncnn::Mat
    ncnn::Mat frame1_mat = avframe_to_ncnn_mat(in_frame);
    if (frame1_mat.empty()) {
        spdlog::error("Failed to convert AVFrame to ncnn::Mat");
        return -1;
    }

    // Allocate an ncnn::Mat for the output frame
    ncnn::Mat output_mat = ncnn::Mat(prev_frame->width, prev_frame->height, (size_t)3, 3);

    // Process the frames using RIFE
    ret = rife->process(frame0_mat, frame1_mat, timestep, output_mat);
    if (ret != 0) {
        spdlog::error("RIFE processing failed");
        return ret;
    }

    // Convert ncnn::Mat to AVFrame
    *out_frame = ncnn_mat_to_avframe(output_mat, output_pix_fmt);

    // Calculate the interpolated PTS
    int64_t delta_pts = in_frame->pts - prev_frame->pts;
    int64_t interpolated_pts = prev_frame->pts + static_cast<int64_t>(delta_pts * timestep + 0.5);

    // Rescale PTS to encoder's time base
    (*out_frame)->pts = av_rescale_q(interpolated_pts, input_time_base, output_time_base);

    // Return the processed frame to the caller
    return ret;
}
