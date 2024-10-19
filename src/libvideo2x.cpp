#include "libvideo2x.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdint>
#include <thread>

#include <spdlog/spdlog.h>
#include <opencv2/videoio.hpp>

#include "decoder.h"
#include "encoder.h"
#include "filter.h"
#include "libplacebo_filter.h"
#include "realesrgan_filter.h"
#include "rife_filter.h"

int process_frames(
    EncoderConfig *encoder_config,
    VideoProcessingContext *proc_ctx,
    AVFormatContext *ifmt_ctx,
    AVFormatContext *ofmt_ctx,
    AVCodecContext *dec_ctx,
    AVCodecContext *enc_ctx,
    Filter *filter,
    int video_stream_index,
    int *stream_mapping,
    bool benchmark = false
) {
    int ret;
    AVPacket packet;
    AVFrame *prev_frame = nullptr;
    std::vector<AVFrame *> flushed_frames;
    char errbuf[AV_ERROR_MAX_STRING_SIZE];

    // Get the total number of frames in the video
    AVStream *video_stream = ifmt_ctx->streams[video_stream_index];
    proc_ctx->total_frames = video_stream->nb_frames;

    // If nb_frames is not set, estimate total frames using duration and frame rate
    if (proc_ctx->total_frames == 0) {
        spdlog::debug("`nb_frames` is not set; estimating total frames with duration*framerate");
        int64_t duration = video_stream->duration;
        AVRational frame_rate = video_stream->avg_frame_rate;
        if (duration != AV_NOPTS_VALUE && frame_rate.num != 0 && frame_rate.den != 0) {
            proc_ctx->total_frames = duration * frame_rate.num / frame_rate.den;
        }
    }

    // If total_frames is still 0, read the total number of frames with OpenCV
    if (proc_ctx->total_frames == 0) {
        spdlog::debug("Unable to estimate total number of frames; reading with OpenCV");
        cv::VideoCapture cap(ifmt_ctx->url);
        if (!cap.isOpened()) {
            spdlog::error("Failed to open video file with OpenCV");
            return -1;
        }
        proc_ctx->total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
        cap.release();
    }

    // Check if the total number of frames is still 0
    if (proc_ctx->total_frames == 0) {
        spdlog::warn("Unable to determine total number of frames");
    } else {
        spdlog::debug("{} frames to process", proc_ctx->total_frames);
    }

    // Get start time
    proc_ctx->start_time = time(NULL);
    if (proc_ctx->start_time == -1) {
        perror("time");
    }

    AVFrame *frame = av_frame_alloc();
    if (frame == nullptr) {
        ret = AVERROR(ENOMEM);
        return ret;
    }

    // Define cleanup function
    auto cleanup = [&]() {
        av_frame_free(&frame);
        if (prev_frame != nullptr) {
            av_frame_free(&prev_frame);
        }
        // Free any flushed frames not yet freed
        for (AVFrame *flushed_frame : flushed_frames) {
            if (flushed_frame) {
                av_frame_free(&flushed_frame);
            }
        }
    };

    // Read frames from the input file
    while (!proc_ctx->abort) {
        ret = av_read_frame(ifmt_ctx, &packet);
        if (ret < 0) {
            if (ret == AVERROR_EOF) {
                spdlog::debug("Reached end of file");
                break;
            }
            av_strerror(ret, errbuf, sizeof(errbuf));
            spdlog::error("Error reading packet: {}", errbuf);
            cleanup();
            return ret;
        }

        if (packet.stream_index == video_stream_index) {
            // Send the packet to the decoder
            ret = avcodec_send_packet(dec_ctx, &packet);
            if (ret < 0) {
                av_strerror(ret, errbuf, sizeof(errbuf));
                spdlog::error("Error sending packet to decoder: {}", errbuf);
                av_packet_unref(&packet);
                cleanup();
                return ret;
            }

            // Receive and process frames from the decoder
            while (!proc_ctx->abort) {
                // Check if the processing is paused
                if (proc_ctx->pause) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }

                ret = avcodec_receive_frame(dec_ctx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    spdlog::debug("Frame not ready");
                    break;
                } else if (ret < 0) {
                    av_strerror(ret, errbuf, sizeof(errbuf));
                    spdlog::error("Error decoding video frame: {}", errbuf);
                    av_packet_unref(&packet);
                    cleanup();
                    return ret;
                }

                if (prev_frame == nullptr) {
                    prev_frame = av_frame_clone(frame);
                }

                // Process the frame using the selected filter
                AVFrame *processed_frame = nullptr;
                switch (filter->get_type()) {
                    case FILTER_OPERATION_UPSCALING:
                        ret = dynamic_cast<UpscalingFilter *>(filter)->upscale(
                            prev_frame, frame, &processed_frame
                        );
                        break;
                    case FILTER_OPERATION_INTERPOLATION:
                        ret = dynamic_cast<InterpolationFilter *>(filter)->interpolate(
                            prev_frame, frame, 0.5f, &processed_frame
                        );
                        break;
                    default:
                        spdlog::error("Unknown filter operation");
                        av_frame_unref(frame);
                        cleanup();
                        return -1;
                }

                if (ret == 0 && processed_frame != nullptr) {
                    // Encode and write the processed frame
                    if (!benchmark) {
                        ret = encode_and_write_frame(
                            processed_frame, enc_ctx, ofmt_ctx, video_stream_index
                        );
                        if (ret < 0) {
                            av_strerror(ret, errbuf, sizeof(errbuf));
                            spdlog::error("Error encoding/writing frame: {}", errbuf);
                            av_frame_free(&processed_frame);
                            av_packet_unref(&packet);
                            av_frame_unref(frame);
                            cleanup();
                            return ret;
                        }

                        if (filter->get_type() == FILTER_OPERATION_INTERPOLATION) {
                            ret = encode_and_write_frame(
                                frame, enc_ctx, ofmt_ctx, video_stream_index
                            );
                            if (ret < 0) {
                                av_strerror(ret, errbuf, sizeof(errbuf));
                                spdlog::error("Error encoding/writing frame: {}", errbuf);
                                av_frame_free(&processed_frame);
                                av_packet_unref(&packet);
                                av_frame_unref(frame);
                                cleanup();
                                return ret;
                            }
                            prev_frame = av_frame_clone(frame);
                        }
                    }

                    av_frame_free(&processed_frame);
                    proc_ctx->processed_frames++;
                } else if (ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
                    spdlog::error("Filter returned an error");
                    av_packet_unref(&packet);
                    av_frame_unref(frame);
                    cleanup();
                    return ret;
                }

                av_frame_unref(frame);
                spdlog::debug(
                    "Processed frame {}/{}", proc_ctx->processed_frames, proc_ctx->total_frames
                );
            }
        } else if (encoder_config->copy_streams && stream_mapping[packet.stream_index] >= 0) {
            AVStream *in_stream = ifmt_ctx->streams[packet.stream_index];
            int out_stream_index = stream_mapping[packet.stream_index];
            AVStream *out_stream = ofmt_ctx->streams[out_stream_index];

            // Rescale packet timestamps
            av_packet_rescale_ts(&packet, in_stream->time_base, out_stream->time_base);
            packet.stream_index = out_stream_index;

            // If copy streams is enabled, copy the packet to the output
            ret = av_interleaved_write_frame(ofmt_ctx, &packet);
            if (ret < 0) {
                av_strerror(ret, errbuf, sizeof(errbuf));
                spdlog::error("Error muxing packet: {}", errbuf);
                av_packet_unref(&packet);
                cleanup();
                return ret;
            }
        }
        av_packet_unref(&packet);
    }

    // Flush the filter
    ret = filter->flush(flushed_frames);
    if (ret < 0) {
        av_strerror(ret, errbuf, sizeof(errbuf));
        spdlog::error("Error flushing filter: {}", errbuf);
        cleanup();
        return ret;
    }

    // Encode and write all flushed frames
    for (AVFrame *&flushed_frame : flushed_frames) {
        ret = encode_and_write_frame(flushed_frame, enc_ctx, ofmt_ctx, video_stream_index);
        if (ret < 0) {
            av_strerror(ret, errbuf, sizeof(errbuf));
            spdlog::error("Error encoding/writing flushed frame: {}", errbuf);
            av_frame_free(&flushed_frame);
            flushed_frame = nullptr;
            cleanup();
            return ret;
        }
        av_frame_free(&flushed_frame);
        flushed_frame = nullptr;
    }

    // Flush the encoder
    ret = flush_encoder(enc_ctx, ofmt_ctx);
    if (ret < 0) {
        av_strerror(ret, errbuf, sizeof(errbuf));
        spdlog::error("Error flushing encoder: {}", errbuf);
        cleanup();
        return ret;
    }

    cleanup();
    return ret;
}

/**
 * @brief Process a video file using the selected filter and encoder settings.
 *
 * @param[in] input_filename Path to the input video file
 * @param[in] output_filename Path to the output video file
 * @param[in] log_level Log level
 * @param[in] benchmark Flag to enable benchmarking mode
 * @param[in] hw_type Hardware device type
 * @param[in] filter_config Filter configurations
 * @param[in] encoder_config Encoder configurations
 * @param[in,out] proc_ctx Video processing context
 * @return int 0 on success, non-zero value on error
 */
extern "C" int process_video(
    const char *input_filename,
    const char *output_filename,
    Libvideo2xLogLevel log_level,
    bool benchmark,
    AVHWDeviceType hw_type,
    const FilterConfig *filter_config,
    EncoderConfig *encoder_config,
    VideoProcessingContext *proc_ctx
) {
    AVFormatContext *ifmt_ctx = nullptr;
    AVFormatContext *ofmt_ctx = nullptr;
    AVCodecContext *dec_ctx = nullptr;
    AVCodecContext *enc_ctx = nullptr;
    AVBufferRef *hw_ctx = nullptr;
    int *stream_mapping = nullptr;
    Filter *filter = nullptr;
    int video_stream_index = -1;
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    int ret = 0;

    // Set the log level for FFmpeg and spdlog (libvideo2x)
    switch (log_level) {
        case LIBVIDEO2X_LOG_LEVEL_TRACE:
            av_log_set_level(AV_LOG_TRACE);
            spdlog::set_level(spdlog::level::trace);
            break;
        case LIBVIDEO2X_LOG_LEVEL_DEBUG:
            av_log_set_level(AV_LOG_DEBUG);
            spdlog::set_level(spdlog::level::debug);
            break;
        case LIBVIDEO2X_LOG_LEVEL_INFO:
            av_log_set_level(AV_LOG_INFO);
            spdlog::set_level(spdlog::level::info);
            break;
        case LIBVIDEO2X_LOG_LEVEL_WARNING:
            av_log_set_level(AV_LOG_WARNING);
            spdlog::set_level(spdlog::level::warn);
            break;
        case LIBVIDEO2X_LOG_LEVEL_ERROR:
            av_log_set_level(AV_LOG_ERROR);
            spdlog::set_level(spdlog::level::err);
            break;
        case LIBVIDEO2X_LOG_LEVEL_CRITICAL:
            av_log_set_level(AV_LOG_FATAL);
            spdlog::set_level(spdlog::level::critical);
            break;
        case LIBVIDEO2X_LOG_LEVEL_OFF:
            av_log_set_level(AV_LOG_QUIET);
            spdlog::set_level(spdlog::level::off);
            break;
        default:
            av_log_set_level(AV_LOG_INFO);
            spdlog::set_level(spdlog::level::info);
            break;
    }

    // Lambda function for cleanup
    auto cleanup = [&]() {
        if (ifmt_ctx) {
            avformat_close_input(&ifmt_ctx);
        }
        if (ofmt_ctx && !(ofmt_ctx->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&ofmt_ctx->pb);
        }
        if (ofmt_ctx) {
            avformat_free_context(ofmt_ctx);
        }
        if (dec_ctx) {
            avcodec_free_context(&dec_ctx);
        }
        if (enc_ctx) {
            avcodec_free_context(&enc_ctx);
        }
        if (hw_ctx) {
            av_buffer_unref(&hw_ctx);
        }
        if (stream_mapping) {
            av_free(stream_mapping);
        }
        if (filter) {
            delete filter;
        }
    };

    // Initialize hardware device context
    if (hw_type != AV_HWDEVICE_TYPE_NONE) {
        ret = av_hwdevice_ctx_create(&hw_ctx, hw_type, NULL, NULL, 0);
        if (ret < 0) {
            av_strerror(ret, errbuf, sizeof(errbuf));
            spdlog::error("Error initializing hardware device context: {}", errbuf);
            cleanup();
            return ret;
        }
    }

    // Initialize input
    ret = init_decoder(hw_type, hw_ctx, input_filename, &ifmt_ctx, &dec_ctx, &video_stream_index);
    if (ret < 0) {
        av_strerror(ret, errbuf, sizeof(errbuf));
        spdlog::error("Failed to initialize decoder: {}", errbuf);
        cleanup();
        return ret;
    }

    // Initialize output based on filter configuration
    int output_width = 0, output_height = 0;
    switch (filter_config->filter_backend) {
        case FILTER_BACKEND_LIBPLACEBO:
            output_width = filter_config->config.libplacebo.output_width;
            output_height = filter_config->config.libplacebo.output_height;
            break;
        case FILTER_BACKEND_REALESRGAN:
            // Calculate the output dimensions based on the scaling factor
            output_width = dec_ctx->width * filter_config->config.realesrgan.scaling_factor;
            output_height = dec_ctx->height * filter_config->config.realesrgan.scaling_factor;
            break;
        case FILTER_BACKEND_RIFE:
            // RIFE does not change the output dimensions
            output_width = dec_ctx->width;
            output_height = dec_ctx->height;
            break;
        default:
            spdlog::error("Unknown filter backend");
            cleanup();
            return -1;
    }
    spdlog::info("Output video dimensions: {}x{}", output_width, output_height);

    // Initialize output encoder
    encoder_config->output_width = output_width;
    encoder_config->output_height = output_height;
    ret = init_encoder(
        hw_ctx,
        output_filename,
        ifmt_ctx,
        &ofmt_ctx,
        &enc_ctx,
        dec_ctx,
        encoder_config,
        video_stream_index,
        &stream_mapping
    );
    if (ret < 0) {
        av_strerror(ret, errbuf, sizeof(errbuf));
        spdlog::error("Failed to initialize encoder: {}", errbuf);
        cleanup();
        return ret;
    }

    // Write the output file header
    ret = avformat_write_header(ofmt_ctx, NULL);
    if (ret < 0) {
        av_strerror(ret, errbuf, sizeof(errbuf));
        spdlog::error("Error occurred when opening output file: {}", errbuf);
        cleanup();
        return ret;
    }

    // Create and initialize the appropriate filter
    switch (filter_config->filter_backend) {
        case FILTER_BACKEND_LIBPLACEBO: {
            const auto &config = filter_config->config.libplacebo;

            // Validate shader path
            if (!config.shader_path) {
                spdlog::error("Shader path must be provided for the libplacebo filter");
                cleanup();
                return -1;
            }

            // Validate output dimensions
            if (config.output_width <= 0 || config.output_height <= 0) {
                spdlog::error("Output dimensions must be provided for the libplacebo filter");
                cleanup();
                return -1;
            }

            filter = new LibplaceboFilter{
                config.output_width, config.output_height, std::filesystem::path(config.shader_path)
            };
            break;
        }
        case FILTER_BACKEND_REALESRGAN: {
            const auto &config = filter_config->config.realesrgan;

            // Validate model name
            if (!config.model) {
                spdlog::error("Model name must be provided for the RealESRGAN filter");
                cleanup();
                return -1;
            }

            // Validate scaling factor
            if (config.scaling_factor <= 0) {
                spdlog::error("Scaling factor must be provided for the RealESRGAN filter");
                cleanup();
                return -1;
            }

            filter = new RealesrganFilter{
                config.gpuid, config.tta_mode, config.scaling_factor, config.model
            };
            break;
        }
        case FILTER_BACKEND_RIFE: {
            const auto &config = filter_config->config.rife;

            // TODO: Validate RIFE configurations
            filter = new RifeFilter{
                config.gpuid,
                config.tta_mode,
                config.tta_temporal_mode,
                config.uhd_mode,
                config.num_threads,
                config.rife_v2,
                config.rife_v4,
                std::filesystem::path(config.model_dir)
            };
            break;
        }
        default:
            spdlog::error("Unknown filter type");
            cleanup();
            return -1;
    }

    // Initialize the filter
    ret = filter->init(dec_ctx, enc_ctx, hw_ctx);
    if (ret < 0) {
        av_strerror(ret, errbuf, sizeof(errbuf));
        spdlog::error("Failed to initialize filter: {}", errbuf);
        cleanup();
        return ret;
    }

    // Process frames
    ret = process_frames(
        encoder_config,
        proc_ctx,
        ifmt_ctx,
        ofmt_ctx,
        dec_ctx,
        enc_ctx,
        filter,
        video_stream_index,
        stream_mapping,
        benchmark
    );
    if (ret < 0) {
        av_strerror(ret, errbuf, sizeof(errbuf));
        spdlog::error("Error processing frames: {}", errbuf);
        cleanup();
        return ret;
    }

    // Write the output file trailer
    av_write_trailer(ofmt_ctx);

    // Cleanup before returning
    cleanup();

    if (ret < 0 && ret != AVERROR_EOF) {
        av_strerror(ret, errbuf, sizeof(errbuf));
        spdlog::error("Error occurred: {}", errbuf);
        return ret;
    }
    return 0;
}
