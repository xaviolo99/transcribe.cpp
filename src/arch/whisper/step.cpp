// arch/whisper/step.cpp - encode / decode steps behind the public
// transcribe_whisper_step_* entry points, for callers that own the decoding
// loop. Runs the same encoder, cross-KV and decoder graphs as whisper_run.

#include "decoder.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "transcribe-debug.h"
#include "transcribe-log.h"
#include "weights.h"
#include "whisper.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace transcribe::whisper {

namespace {

// A step graph's self-attention window. Small enough for the common pass, and
// rebuilt at the decoder's full context when a prompt outgrows it.
constexpr int k_step_window = 256;
constexpr int k_step_margin = 64;

bool primary_is_gpu(const WhisperModel & cm) {
    return cm.plan.primary_kind != transcribe::BackendKind::Cpu &&
           cm.plan.primary_kind != transcribe::BackendKind::Accel &&
           cm.plan.primary_kind != transcribe::BackendKind::Unknown;
}

bool ensure_kv_cache(WhisperSession * cc, WhisperModel * cm, int T_enc) {
    const int n_ctx = cm->hparams.dec_max_target_positions;
    if (cc->kv_cache.ctx != nullptr && cc->kv_cache.n_batch == 1 && cc->kv_cache.n_ctx == n_ctx &&
        cc->kv_cache.T_enc == T_enc) {
        return true;
    }
    cc->kv_cache.free();
    ggml_type kv_type = GGML_TYPE_F16;
    if (cc->kv_type == TRANSCRIBE_KV_TYPE_F32) {
        kv_type = GGML_TYPE_F32;
    } else if (cc->kv_type != TRANSCRIBE_KV_TYPE_F16) {
        const ggml_tensor * probe = !cm->weights.dec_blocks.empty() ? cm->weights.dec_blocks[0].self_q_w : nullptr;
        kv_type                   = (probe != nullptr && probe->type == GGML_TYPE_F32) ? GGML_TYPE_F32 : GGML_TYPE_F16;
    }
    return kv_cache_init(cc->kv_cache, cm->plan.primary, n_ctx, T_enc, cm->hparams.dec_d_model,
                         cm->hparams.dec_n_layers, kv_type);
}

transcribe_status compute_cross_kv(WhisperSession * cc, WhisperModel * cm, int T_enc) {
    if (!step_ensure_compute_ctx(cc, 8 * 1024 * 1024)) {
        return TRANSCRIBE_ERR_OOM;
    }
    DecoderBuild db =
        build_cross_kv_graph(cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache, cc->enc_out.tensor, T_enc);
    if (db.graph == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_sched_reset(cc->sched);
    if (!TRANSCRIBE_ALLOC_SCHED_GRAPH(cc->sched, db.graph, "whisper")) {
        return TRANSCRIBE_ERR_OOM;
    }
    if (ggml_backend_sched_graph_compute(cc->sched, db.graph) != GGML_STATUS_SUCCESS) {
        return TRANSCRIBE_ERR_BACKEND;
    }
    cc->kv_cache.cross_populated = true;
    return TRANSCRIBE_OK;
}

// Upload the padded-slot mask of the cross cache: 0 over the T_enc real frames,
// -inf past them, one row per token.
void set_cross_mask(ggml_tensor * mask_in, int T_enc, int n_rows) {
    if (mask_in == nullptr) {
        return;
    }
    const int          n_kv = static_cast<int>(mask_in->ne[0]);
    std::vector<float> mask(static_cast<size_t>(n_kv) * n_rows, 0.0f);
    for (int q = 0; q < n_rows; ++q) {
        for (int k = T_enc; k < n_kv; ++k) {
            mask[static_cast<size_t>(q) * n_kv + k] = -1e9f;
        }
    }
    ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(float));
}

transcribe_status decode_prompt(WhisperSession * cc,
                                WhisperModel *   cm,
                                const int32_t *  tokens,
                                int              n_tokens,
                                int              n_past,
                                const int32_t *  logit_rows,
                                int              n_logit_rows) {
    WhisperStepState & st = cc->step_api;
    st.invalidate_graphs();
    if (!step_ensure_compute_ctx(cc, 16 * 1024 * 1024)) {
        return TRANSCRIBE_ERR_OOM;
    }
    const int    T_enc  = cc->kv_cache.T_enc;
    const int    kv_pad = kv_pad_self_attn(cm->plan.primary_kind, cc->decoder_use_flash);
    DecoderBuild db =
        build_decoder_graph_kv(cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache, n_tokens, n_past, T_enc, kv_pad,
                               /*skip_log_softmax=*/true, cc->decoder_use_flash, n_logit_rows, &st.capture);
    if (db.out == nullptr || db.graph == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_sched_reset(cc->sched);
    if (!TRANSCRIBE_ALLOC_SCHED_GRAPH(cc->sched, db.graph, "whisper")) {
        return TRANSCRIBE_ERR_OOM;
    }

    ggml_backend_tensor_set(db.token_ids_in, tokens, 0, static_cast<size_t>(n_tokens) * sizeof(int32_t));
    std::vector<int32_t> pos_ids(static_cast<size_t>(n_tokens));
    for (int i = 0; i < n_tokens; ++i) {
        pos_ids[static_cast<size_t>(i)] = n_past + i;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(db.graph, "dec.pos_ids"), pos_ids.data(), 0,
                            pos_ids.size() * sizeof(int32_t));
    if (db.logit_rows_in != nullptr) {
        ggml_backend_tensor_set(db.logit_rows_in, logit_rows, 0, static_cast<size_t>(n_logit_rows) * sizeof(int32_t));
    }
    if (db.causal_mask_in != nullptr) {
        const int          n_kv = static_cast<int>(db.causal_mask_in->ne[0]);
        std::vector<float> mask(static_cast<size_t>(n_kv) * n_tokens);
        for (int q = 0; q < n_tokens; ++q) {
            for (int k = 0; k < n_kv; ++k) {
                mask[static_cast<size_t>(q) * n_kv + k] = k <= n_past + q ? 0.0f : -1e9f;
            }
        }
        ggml_backend_tensor_set(db.causal_mask_in, mask.data(), 0, mask.size() * sizeof(float));
    }
    set_cross_mask(db.cross_mask_in, T_enc, n_tokens);

    if (ggml_backend_sched_graph_compute(cc->sched, db.graph) != GGML_STATUS_SUCCESS) {
        return TRANSCRIBE_ERR_BACKEND;
    }
    st.logits       = db.out;
    st.n_logit_rows = n_logit_rows > 0 ? n_logit_rows : n_tokens;
    st.n_attn_rows  = n_tokens;
    return TRANSCRIBE_OK;
}

transcribe_status decode_one(WhisperSession * cc, WhisperModel * cm, int32_t token, int n_past) {
    WhisperStepState & st         = cc->step_api;
    const int          n_ctx      = cm->hparams.dec_max_target_positions;
    const ggml_fp16_t  mask_zero  = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t  mask_block = ggml_fp32_to_fp16(-INFINITY);

    if (st.step.graph == nullptr || n_past + 1 > st.step.max_n_kv) {
        const int max_n_kv = n_past + k_step_margin <= k_step_window ? std::min(k_step_window, n_ctx) : n_ctx;
        st.invalidate_graphs();
        if (!step_ensure_compute_ctx(cc, 8 * 1024 * 1024)) {
            return TRANSCRIBE_ERR_OOM;
        }
        StepBuild sb = build_step_graph(cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache, max_n_kv,
                                        cc->kv_cache.T_enc, cc->decoder_use_flash, &st.capture);
        if (sb.graph == nullptr || sb.logits_out == nullptr) {
            return TRANSCRIBE_ERR_GGUF;
        }
        ggml_backend_sched_reset(cc->sched);
        if (!TRANSCRIBE_ALLOC_SCHED_GRAPH(cc->sched, sb.graph, "whisper")) {
            return TRANSCRIBE_ERR_OOM;
        }
        set_cross_mask(sb.cross_mask_in, cc->kv_cache.T_enc, 1);
        st.step        = sb;
        st.step_mask_n = -1;
    }

    if (st.step_mask_n != n_past) {
        st.step_mask.assign(static_cast<size_t>(st.step.max_n_kv), mask_block);
        std::fill_n(st.step_mask.begin(), n_past, mask_zero);
    }
    st.step_mask[static_cast<size_t>(n_past)] = mask_zero;
    st.step_mask_n                            = n_past + 1;

    const int32_t pos    = n_past;
    const int64_t kv_idx = n_past;
    ggml_backend_tensor_set(st.step.token_id_in, &token, 0, sizeof(int32_t));
    ggml_backend_tensor_set(st.step.pos_id_in, &pos, 0, sizeof(int32_t));
    ggml_backend_tensor_set(st.step.kv_idx_in, &kv_idx, 0, sizeof(int64_t));
    ggml_backend_tensor_set(st.step.mask_in, st.step_mask.data(), 0, st.step_mask.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_sched_graph_compute(cc->sched, st.step.graph) != GGML_STATUS_SUCCESS) {
        return TRANSCRIBE_ERR_BACKEND;
    }
    st.logits       = st.step.logits_out;
    st.n_logit_rows = 1;
    st.n_attn_rows  = 1;
    return TRANSCRIBE_OK;
}

}  // namespace

transcribe_status step_set_align_heads(WhisperSession * cc, const int32_t * layer_head_pairs, int n_heads) {
    if (n_heads < 0 || (n_heads > 0 && layer_head_pairs == nullptr)) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const auto * cm = static_cast<const WhisperModel *>(cc->model);

    std::vector<WhisperAlignHead> heads(static_cast<size_t>(n_heads));
    for (int i = 0; i < n_heads; ++i) {
        heads[static_cast<size_t>(i)].layer = layer_head_pairs[2 * i];
        heads[static_cast<size_t>(i)].head  = layer_head_pairs[2 * i + 1];
        if (heads[static_cast<size_t>(i)].layer < 0 ||
            heads[static_cast<size_t>(i)].layer >= cm->hparams.dec_n_layers || heads[static_cast<size_t>(i)].head < 0 ||
            heads[static_cast<size_t>(i)].head >= cm->hparams.dec_n_heads) {
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
    }
    WhisperStepState & st = cc->step_api;
    st.invalidate_graphs();
    st.align_heads   = std::move(heads);
    st.capture.heads = st.align_heads.empty() ? nullptr : &st.align_heads;
    return TRANSCRIBE_OK;
}

transcribe_status step_encode(WhisperSession * cc, const float * pcm, int n_samples) {
    if (pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto * cm = static_cast<WhisperModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (!cm->mel.has_value()) {
        return TRANSCRIBE_ERR_GGUF;
    }
    transcribe::debug::init();

    WhisperStepState & st = cc->step_api;
    st.invalidate_graphs();
    st.encoded = false;

    const int n_mels       = cm->hparams.enc_num_mel_bins;
    const int n_mel_frames = cm->hparams.fe_nb_max_frames > 0 ? cm->hparams.fe_nb_max_frames : 3000;
    const int n_window     = cm->hparams.fe_n_samples > 0 ? cm->hparams.fe_n_samples : 480000;

    // What openai's log_mel_spectrogram(padding=N_SAMPLES) + pad_or_trim feeds the
    // encoder: the audio zero-padded to the window, first frames kept. The STFT
    // only runs over the audio and enough zeros for the last frame it emits to see
    // no audio; every later frame is the same, so it repeats that one.
    const int64_t      t_mel_start = ggml_time_us();
    const int          n_tail      = cm->hparams.fe_n_fft + cm->hparams.fe_hop_length;
    std::vector<float> pcm_work;
    const float *      pcm_in   = pcm;
    size_t             pcm_in_n = static_cast<size_t>(n_samples);
    if (n_samples < n_window) {
        pcm_in_n = static_cast<size_t>(std::min(n_samples + n_tail, n_window));
        pcm_work.assign(pcm_in_n, 0.0f);
        std::memcpy(pcm_work.data(), pcm, static_cast<size_t>(n_samples) * sizeof(float));
        pcm_in = pcm_work.data();
    }
    int                mel_n_mels = 0, mel_n_frames = 0;
    std::vector<float> mel_mn;  // [n_mels, n_frames] (MelFrontend layout)
    if (const transcribe_status mst =
            cm->mel->compute(pcm_in, pcm_in_n, mel_mn, mel_n_mels, mel_n_frames, cc->n_threads);
        mst != TRANSCRIBE_OK) {
        return mst;
    }
    if (mel_n_mels != n_mels || mel_n_frames < 1) {
        return TRANSCRIBE_ERR_GGUF;
    }
    cc->mel_buf.resize(static_cast<size_t>(n_mels) * static_cast<size_t>(n_mel_frames));
    for (int t = 0; t < n_mel_frames; ++t) {
        const int src = std::min(t, mel_n_frames - 1);
        for (int m = 0; m < n_mels; ++m) {
            cc->mel_buf[static_cast<size_t>(t) * n_mels + m] = mel_mn[static_cast<size_t>(m) * mel_n_frames + src];
        }
    }
    cc->t_mel_us = ggml_time_us() - t_mel_start;

    const int64_t t_encode_start = ggml_time_us();
    int           T_enc          = 0;
    if (const transcribe_status est = step_run_encoder(cc, cm, cc->mel_buf.data(), n_mels, n_mel_frames, T_enc);
        est != TRANSCRIBE_OK) {
        return est;
    }
    if (!ensure_kv_cache(cc, cm, T_enc)) {
        return TRANSCRIBE_ERR_OOM;
    }
    if (const transcribe_status kst = compute_cross_kv(cc, cm, T_enc); kst != TRANSCRIBE_OK) {
        return kst;
    }
    cc->t_encode_us   = ggml_time_us() - t_encode_start;
    cc->kv_cache.n    = 0;
    cc->kv_cache.head = 0;
    st.encoded        = true;
    return TRANSCRIBE_OK;
}

transcribe_status step_decode(WhisperSession * cc,
                              const int32_t *  tokens,
                              int              n_tokens,
                              int              n_past,
                              const int32_t *  logit_rows,
                              int              n_logit_rows) {
    auto *             cm = static_cast<WhisperModel *>(cc->model);
    WhisperStepState & st = cc->step_api;
    if (!st.encoded || tokens == nullptr || n_tokens <= 0 || n_past < 0 || n_past > cc->kv_cache.n ||
        n_logit_rows < 0 || (n_logit_rows > 0 && logit_rows == nullptr)) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (n_past + n_tokens > cc->kv_cache.n_ctx) {
        return TRANSCRIBE_ERR_INPUT_TOO_LONG;
    }
    for (int i = 0; i < n_tokens; ++i) {
        if (tokens[i] < 0 || tokens[i] >= cm->hparams.dec_vocab_size) {
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
    }
    for (int i = 0; i < n_logit_rows; ++i) {
        if (logit_rows[i] < 0 || logit_rows[i] >= n_tokens) {
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
    }

    // Steady-state steps go through the reusable single-token graph where
    // whisper_run does; prompts rebuild per call since their length varies.
    const bool one_step =
        n_tokens == 1 && n_past > 0 && n_logit_rows == 0 && primary_is_gpu(*cm) && !transcribe::debug::enabled();
    const transcribe_status status = one_step ?
                                         decode_one(cc, cm, tokens[0], n_past) :
                                         decode_prompt(cc, cm, tokens, n_tokens, n_past, logit_rows, n_logit_rows);
    if (status != TRANSCRIBE_OK) {
        st.invalidate_graphs();
        return status;
    }
    cc->kv_cache.n    = n_past + n_tokens;
    cc->kv_cache.head = n_past + n_tokens;
    return TRANSCRIBE_OK;
}

transcribe_status step_get_logits(const WhisperSession * cc, int row, float * out, size_t n_out) {
    const auto *             cm      = static_cast<const WhisperModel *>(cc->model);
    const WhisperStepState & st      = cc->step_api;
    const size_t             n_vocab = static_cast<size_t>(cm->hparams.dec_vocab_size);
    if (st.logits == nullptr || out == nullptr || row < 0 || row >= st.n_logit_rows || n_out < n_vocab) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    ggml_backend_tensor_get(st.logits, out, static_cast<size_t>(row) * n_vocab * sizeof(float),
                            n_vocab * sizeof(float));
    return TRANSCRIBE_OK;
}

transcribe_status step_get_cross_attention(const WhisperSession * cc, float * out, size_t n_out) {
    const WhisperStepState & st      = cc->step_api;
    const size_t             n_heads = st.align_heads.size();
    if (st.logits == nullptr || out == nullptr || st.capture.weights.size() != n_heads) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const size_t n_each = static_cast<size_t>(st.n_attn_rows) * static_cast<size_t>(cc->kv_cache.T_enc);
    if (n_out < n_heads * n_each) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    for (size_t i = 0; i < n_heads; ++i) {
        ggml_backend_tensor_get(st.capture.weights[i], out + i * n_each, 0, n_each * sizeof(float));
    }
    return TRANSCRIBE_OK;
}

}  // namespace transcribe::whisper
