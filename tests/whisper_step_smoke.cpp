// whisper_step_smoke.cpp - real-model gated public-ABI test for the Whisper step
// API: a caller-owned greedy loop reproduces transcribe_run's text, a prompt
// decoded whole or token by token gives the same logits, the skipped padding of
// the mel leaves the output unchanged, and the alignment heads' attention is a
// distribution over frames that walks forward through the audio.

#include "transcribe.h"
#include "transcribe/whisper.h"
#include "wav.h"

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifndef TRANSCRIBE_TEST_SAMPLES_DIR
#    define TRANSCRIBE_TEST_SAMPLES_DIR "samples"
#endif

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

// The multilingual token table (large-v3's extra language shifts two more below).
constexpr int32_t k_eot           = 50257;
constexpr int32_t k_sot           = 50258;
constexpr int32_t k_lang_en       = 50259;
constexpr int32_t k_first_special = 50257;

int argmax(const std::vector<float> & v, int n) {
    return static_cast<int>(std::max_element(v.begin(), v.begin() + n) - v.begin());
}

}  // namespace

int main() {
    const char * env = std::getenv("TRANSCRIBE_WHISPER_GGUF");
    if (env == nullptr || env[0] == '\0') {
        std::fprintf(stderr, "whisper_step_smoke: TRANSCRIBE_WHISPER_GGUF not set; skipping.\n");
        return 77;
    }
    struct stat st{};
    if (::stat(env, &st) != 0) {
        std::fprintf(stderr, "whisper_step_smoke: model not found: %s\n", env);
        return 77;
    }

    const std::string  wav_path = std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/jfk.wav";
    std::vector<float> pcm;
    std::string        wav_err;
    if (!transcribe_cli::load_wav_mono_16k(wav_path, pcm, wav_err)) {
        std::fprintf(stderr, "whisper_step_smoke: wav load: %s\n", wav_err.c_str());
        return EXIT_FAILURE;
    }

    transcribe_model_load_params load_params;
    transcribe_model_load_params_init(&load_params);
    transcribe_model * model = nullptr;
    CHECK(transcribe_model_load_file(env, &load_params, &model) == TRANSCRIBE_OK);

    transcribe_session_params session_params;
    transcribe_session_params_init(&session_params);
    transcribe_session * session = nullptr;
    CHECK(transcribe_session_init(model, &session_params, &session) == TRANSCRIBE_OK);

    transcribe_whisper_dims dims;
    transcribe_whisper_dims_init(&dims);
    CHECK(transcribe_whisper_get_dims(model, &dims) == TRANSCRIBE_OK);
    CHECK(dims.n_audio_ctx == 1500);
    if (dims.n_vocab < 51865) {
        std::fprintf(stderr, "whisper_step_smoke: needs a multilingual model; skipping.\n");
        transcribe_session_free(session);
        transcribe_model_free(model);
        return 77;
    }
    const int32_t k_transcribe   = dims.n_vocab >= 51866 ? 50360 : 50359;
    const int32_t k_notimestamps = dims.n_vocab >= 51866 ? 50364 : 50363;

    // Before an encode there is nothing to decode against, and no session is an error.
    const int32_t prompt[] = { k_sot, k_lang_en, k_transcribe, k_notimestamps };
    CHECK(transcribe_whisper_step_decode(nullptr, prompt, 4, 0, nullptr, 0) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_whisper_step_decode(session, prompt, 4, 0, nullptr, 0) == TRANSCRIBE_ERR_INVALID_ARG);

    // Align heads are validated against the model.
    const int32_t bad_heads[] = { dims.n_text_layer, 0 };
    CHECK(transcribe_whisper_step_set_align_heads(session, bad_heads, 1) == TRANSCRIBE_ERR_INVALID_ARG);
    const int32_t twice[] = { 0, 0, 0, 0 };
    CHECK(transcribe_whisper_step_set_align_heads(session, twice, 2) == TRANSCRIBE_ERR_INVALID_ARG);
    const int32_t heads[] = { dims.n_text_layer - 1, 0, dims.n_text_layer - 1, 1 };
    const int     n_heads = 2;
    CHECK(transcribe_whisper_step_set_align_heads(session, heads, n_heads) == TRANSCRIBE_OK);

    const int n_samples = static_cast<int>(pcm.size());
    {
        // One bad sample is refused, not turned into NaN logits for the whole window.
        std::vector<float> bad = pcm;
        bad[100]               = std::nanf("");
        CHECK(transcribe_whisper_step_encode(session, bad.data(), n_samples) == TRANSCRIBE_ERR_INVALID_ARG);
    }
    CHECK(transcribe_whisper_step_encode(session, pcm.data(), n_samples) == TRANSCRIBE_OK);

    const int          n_vocab  = dims.n_vocab;
    const int          n_frames = dims.n_audio_ctx;
    std::vector<float> logits(static_cast<size_t>(n_vocab));
    std::vector<float> attn(static_cast<size_t>(n_heads) * n_frames);

    // A prompt asking for its last row.
    const int32_t last_row = 3;
    CHECK(transcribe_whisper_step_decode(session, prompt, 4, 0, &last_row, 1) == TRANSCRIBE_OK);
    CHECK(transcribe_whisper_step_get_logits(session, 0, logits.data(), logits.size()) == TRANSCRIBE_OK);
    CHECK(transcribe_whisper_step_get_logits(session, 1, logits.data(), logits.size()) == TRANSCRIBE_ERR_INVALID_ARG);
    std::vector<float> first_logits = logits;

    // The same prompt token by token.
    CHECK(transcribe_whisper_step_decode(session, prompt, 1, 0, nullptr, 0) == TRANSCRIBE_OK);
    for (int i = 1; i < 4; ++i) {
        CHECK(transcribe_whisper_step_decode(session, prompt + i, 1, i, nullptr, 0) == TRANSCRIBE_OK);
    }
    CHECK(transcribe_whisper_step_get_logits(session, 0, logits.data(), logits.size()) == TRANSCRIBE_OK);
    float max_diff = 0.0f;
    for (int i = 0; i < n_vocab; ++i) {
        max_diff = std::max(max_diff, std::fabs(logits[static_cast<size_t>(i)] - first_logits[static_cast<size_t>(i)]));
    }
    // Not bit-equal: flash attention accumulates in F16 and rows arrive in a
    // different order.
    CHECK(max_diff < 0.3f);
    CHECK(argmax(logits, n_vocab) == argmax(first_logits, n_vocab));

    // Rewinding and redoing a token gives exactly what it gave the first time, and an
    // impossible length is an error even where the addition would overflow.
    {
        std::vector<float> again(logits.size());
        CHECK(transcribe_whisper_step_decode(session, prompt + 3, 1, 3, nullptr, 0) == TRANSCRIBE_OK);
        CHECK(transcribe_whisper_step_get_logits(session, 0, again.data(), again.size()) == TRANSCRIBE_OK);
        CHECK(again == logits);
        CHECK(transcribe_whisper_step_decode(session, prompt, INT32_MAX, 4, nullptr, 0) ==
              TRANSCRIBE_ERR_INPUT_TOO_LONG);
    }

    // Skipping the window's padding in the mel changes nothing.
    {
        std::vector<float> padded = pcm;
        padded.resize(480000, 0.0f);
        CHECK(transcribe_whisper_step_encode(session, padded.data(), 480000) == TRANSCRIBE_OK);
        CHECK(transcribe_whisper_step_decode(session, prompt, 4, 0, &last_row, 1) == TRANSCRIBE_OK);
        CHECK(transcribe_whisper_step_get_logits(session, 0, logits.data(), logits.size()) == TRANSCRIBE_OK);
        for (int i = 0; i < n_vocab; ++i) {
            CHECK(logits[static_cast<size_t>(i)] == first_logits[static_cast<size_t>(i)]);
        }
    }

    // A caller-owned greedy loop.
    CHECK(transcribe_whisper_step_encode(session, pcm.data(), n_samples) == TRANSCRIBE_OK);
    CHECK(transcribe_whisper_step_decode(session, prompt, 4, 0, &last_row, 1) == TRANSCRIBE_OK);
    CHECK(transcribe_whisper_step_get_logits(session, 0, logits.data(), logits.size()) == TRANSCRIBE_OK);
    std::vector<int32_t> out;
    int                  n_past     = 4;
    int                  last_frame = -1;
    int                  n_forward  = 0;
    logits[220]                     = -INFINITY;  // begin_suppress_tokens: a lone space, EOT
    logits[k_eot]                   = -INFINITY;
    for (int step = 0; step < 100; ++step) {
        const int32_t next = argmax(logits, k_first_special + 1);
        if (next == k_eot) {
            break;
        }
        out.push_back(next);
        CHECK(transcribe_whisper_step_decode(session, &next, 1, n_past, nullptr, 0) == TRANSCRIBE_OK);
        ++n_past;
        CHECK(transcribe_whisper_step_get_logits(session, 0, logits.data(), logits.size()) == TRANSCRIBE_OK);
        CHECK(transcribe_whisper_step_get_cross_attention(session, attn.data(), attn.size()) == TRANSCRIBE_OK);
        for (int h = 0; h < n_heads; ++h) {
            double sum = 0.0;
            for (int f = 0; f < n_frames; ++f) {
                sum += attn[static_cast<size_t>(h) * n_frames + f];
            }
            CHECK(std::fabs(sum - 1.0) < 1e-3);
        }
        const int frame = argmax(attn, n_frames);
        n_forward += frame >= last_frame ? 1 : 0;
        last_frame = frame;
    }
    CHECK(!out.empty());
    CHECK(n_forward >= static_cast<int>(out.size()) / 2);

    transcribe_run_params run_params;
    transcribe_run_params_init(&run_params);
    run_params.language   = "en";
    run_params.timestamps = TRANSCRIBE_TIMESTAMPS_NONE;
    CHECK(transcribe_run(session, pcm.data(), n_samples, &run_params) == TRANSCRIBE_OK);
    const std::string    run_text = std::string(" ") + transcribe_full_text(session);
    std::vector<int32_t> run_ids(1024);
    const int            n_run = transcribe_tokenize(model, run_text.c_str(), run_ids.data(), run_ids.size());
    CHECK(n_run == static_cast<int>(out.size()));
    if (n_run == static_cast<int>(out.size())) {
        CHECK(std::equal(out.begin(), out.end(), run_ids.begin()));
    }

    // A run retires the encoded window.
    CHECK(transcribe_whisper_step_decode(session, prompt, 4, 0, nullptr, 0) == TRANSCRIBE_ERR_INVALID_ARG);

    transcribe_session_free(session);
    transcribe_model_free(model);
    if (g_failures != 0) {
        std::fprintf(stderr, "whisper_step_smoke: %d failure(s)\n", g_failures);
        return 1;
    }
    return 0;
}
