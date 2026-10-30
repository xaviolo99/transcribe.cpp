// arch/whisper/public.cpp - Whisper-family public C entry points.
//
// The run-extension + chunk-trace init and accessor functions live here, not
// in the generic transcribe.cpp, so the central dispatcher stays family-
// agnostic (matching parakeet/moonshine). Public ABI from
// include/transcribe/whisper.h.

#include "transcribe-abi.h"    // check_struct_size, copy_out_prefix
#include "transcribe-api-guard.h"
#include "transcribe-arch.h"   // Arch (session->model->arch)
#include "transcribe-model.h"  // transcribe_model (session->model)
#include "whisper.h"           // WhisperSession (+ transcribe/whisper.h, transcribe-session.h)

#include <cstddef>
#include <cstring>

using transcribe::check_struct_size;
using transcribe::copy_out_prefix;

namespace {

// The library relies on the prefix up to and including n_fallbacks (the
// last field). Matches the central dispatcher's TRANSCRIBE_FIELD_END rule.
constexpr size_t k_min_whisper_chunk_trace_size = offsetof(struct transcribe_whisper_chunk_trace, n_fallbacks) +
                                                  sizeof(((struct transcribe_whisper_chunk_trace *) 0)->n_fallbacks);

// Whisper-specific decoding trace storage lives on WhisperSession; we
// downcast via the model's arch name rather than RTTI so non-RTTI builds
// keep working. Non-Whisper sessions and out-of-range indices are defined
// as zeroed/empty returns.
const transcribe::whisper::WhisperSession * maybe_whisper_context(const struct transcribe_session * session) {
    if (session == nullptr) {
        return nullptr;
    }
    if (session->model == nullptr) {
        return nullptr;
    }
    if (session->model->arch == nullptr) {
        return nullptr;
    }
    const char * name = session->model->arch->name;
    if (name == nullptr || std::strcmp(name, "whisper") != 0) {
        return nullptr;
    }
    return static_cast<const transcribe::whisper::WhisperSession *>(session);
}

transcribe::whisper::WhisperSession * maybe_whisper_context(struct transcribe_session * session) {
    return const_cast<transcribe::whisper::WhisperSession *>(
        maybe_whisper_context(static_cast<const struct transcribe_session *>(session)));
}

}  // namespace

// Init functions (single source of truth for Whisper struct defaults).
extern "C" void transcribe_whisper_chunk_trace_init(struct transcribe_whisper_chunk_trace * p) {
    if (p == nullptr) {
        return;
    }
    std::memset(p, 0, sizeof(*p));
    p->struct_size = sizeof(*p);
}

extern "C" void transcribe_whisper_run_ext_init(struct transcribe_whisper_run_ext * p) {
    if (p == nullptr) {
        return;
    }
    std::memset(p, 0, sizeof(*p));
    p->ext.size                = sizeof(*p);
    p->ext.kind                = TRANSCRIBE_EXT_KIND_WHISPER_RUN;
    // Non-zero recipe defaults (zero-valued fields covered by memset:
    // initial_prompt/prompt_tokens NULL, n_prompt_tokens 0,
    // condition_on_prev_tokens false, temperature 0.0, seed 0).
    p->prompt_condition        = TRANSCRIBE_WHISPER_PROMPT_FIRST_SEGMENT;
    p->max_prev_context_tokens = 223;
    p->temperature_inc         = 0.2f;
    p->compression_ratio_thold = 2.4f;
    p->logprob_thold           = -1.0f;
    p->no_speech_thold         = 0.6f;
    p->max_initial_timestamp   = 1.0f;
}

// Chunk-trace accessors.
extern "C" int transcribe_get_whisper_chunk_count(const struct transcribe_session * session) {
    const auto * wc = maybe_whisper_context(session);
    if (wc == nullptr) {
        return 0;
    }
    return static_cast<int>(wc->chunk_traces.size());
}

extern "C" transcribe_status transcribe_get_whisper_chunk_trace(const struct transcribe_session *       session,
                                                                int                                     i,
                                                                struct transcribe_whisper_chunk_trace * out_trace) {
    if (out_trace == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (const auto st = check_struct_size(out_trace->struct_size, k_min_whisper_chunk_trace_size);
        st != TRANSCRIBE_OK) {
        return st;
    }
    const uint64_t                 caller_size = out_trace->struct_size;
    transcribe_whisper_chunk_trace zero{};
    zero.struct_size = caller_size;
    copy_out_prefix(out_trace, &zero, caller_size, sizeof(zero));
    const auto * wc = maybe_whisper_context(session);
    if (wc == nullptr) {
        return TRANSCRIBE_OK;
    }
    if (i < 0 || static_cast<size_t>(i) >= wc->chunk_traces.size()) {
        return TRANSCRIBE_OK;
    }
    transcribe_whisper_chunk_trace staged = wc->chunk_traces[static_cast<size_t>(i)];
    staged.struct_size                    = caller_size;
    copy_out_prefix(out_trace, &staged, caller_size, sizeof(staged));
    return TRANSCRIBE_OK;
}

// Step API.
extern "C" void transcribe_whisper_dims_init(struct transcribe_whisper_dims * out) {
    if (out == nullptr) {
        return;
    }
    std::memset(out, 0, sizeof(*out));
    out->struct_size = sizeof(*out);
}

extern "C" transcribe_status transcribe_whisper_get_dims(const struct transcribe_model *  model,
                                                         struct transcribe_whisper_dims * out) {
    if (model == nullptr || out == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    constexpr size_t k_min_size = TRANSCRIBE_FIELD_END(struct transcribe_whisper_dims, n_text_layer);
    if (const auto st = check_struct_size(out->struct_size, k_min_size); st != TRANSCRIBE_OK) {
        return st;
    }
    if (model->arch == nullptr || model->arch->name == nullptr || std::strcmp(model->arch->name, "whisper") != 0) {
        return TRANSCRIBE_ERR_UNSUPPORTED_ARCH;
    }
    const auto &                   hp = static_cast<const transcribe::whisper::WhisperModel *>(model)->hparams;
    struct transcribe_whisper_dims staged{};
    staged.n_mels        = hp.enc_num_mel_bins;
    staged.n_audio_ctx   = hp.enc_max_source_positions;
    staged.n_audio_state = hp.enc_d_model;
    staged.n_audio_head  = hp.enc_n_heads;
    staged.n_audio_layer = hp.enc_n_layers;
    staged.n_vocab       = hp.dec_vocab_size;
    staged.n_text_ctx    = hp.dec_max_target_positions;
    staged.n_text_state  = hp.dec_d_model;
    staged.n_text_head   = hp.dec_n_heads;
    staged.n_text_layer  = hp.dec_n_layers;
    staged.struct_size   = out->struct_size;
    copy_out_prefix(out, &staged, out->struct_size, sizeof(staged));
    return TRANSCRIBE_OK;
}

extern "C" transcribe_status transcribe_whisper_step_set_align_heads(struct transcribe_session * session,
                                                                     const int32_t *             layer_head_pairs,
                                                                     int                         n_heads) {
    return transcribe::api_guard_status("transcribe_whisper_step_set_align_heads", [&]() -> transcribe_status {
        if (session == nullptr) {
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
        auto * wc = maybe_whisper_context(session);
        return wc == nullptr ? TRANSCRIBE_ERR_UNSUPPORTED_ARCH :
                               transcribe::whisper::step_set_align_heads(wc, layer_head_pairs, n_heads);
    });
}

extern "C" transcribe_status transcribe_whisper_step_encode(struct transcribe_session * session,
                                                            const float *               pcm,
                                                            int                         n_samples) {
    return transcribe::api_guard_status("transcribe_whisper_step_encode", [&]() -> transcribe_status {
        if (session == nullptr) {
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
        auto * wc = maybe_whisper_context(session);
        return wc == nullptr ? TRANSCRIBE_ERR_UNSUPPORTED_ARCH : transcribe::whisper::step_encode(wc, pcm, n_samples);
    });
}

extern "C" transcribe_status transcribe_whisper_step_decode(struct transcribe_session * session,
                                                            const int32_t *             tokens,
                                                            int                         n_tokens,
                                                            int                         n_past,
                                                            const int32_t *             logit_rows,
                                                            int                         n_logit_rows) {
    return transcribe::api_guard_status("transcribe_whisper_step_decode", [&]() -> transcribe_status {
        if (session == nullptr) {
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
        auto * wc = maybe_whisper_context(session);
        return wc == nullptr ? TRANSCRIBE_ERR_UNSUPPORTED_ARCH :
                               transcribe::whisper::step_decode(wc, tokens, n_tokens, n_past, logit_rows, n_logit_rows);
    });
}

extern "C" transcribe_status transcribe_whisper_step_get_logits(const struct transcribe_session * session,
                                                                int                               row,
                                                                float *                           out,
                                                                size_t                            n_out) {
    return transcribe::api_guard_status("transcribe_whisper_step_get_logits", [&]() -> transcribe_status {
        if (session == nullptr) {
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
        const auto * wc = maybe_whisper_context(session);
        return wc == nullptr ? TRANSCRIBE_ERR_UNSUPPORTED_ARCH :
                               transcribe::whisper::step_get_logits(wc, row, out, n_out);
    });
}

extern "C" transcribe_status transcribe_whisper_step_get_cross_attention(const struct transcribe_session * session,
                                                                         float *                           out,
                                                                         size_t                            n_out) {
    return transcribe::api_guard_status("transcribe_whisper_step_get_cross_attention", [&]() -> transcribe_status {
        if (session == nullptr) {
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
        const auto * wc = maybe_whisper_context(session);
        return wc == nullptr ? TRANSCRIBE_ERR_UNSUPPORTED_ARCH :
                               transcribe::whisper::step_get_cross_attention(wc, out, n_out);
    });
}
