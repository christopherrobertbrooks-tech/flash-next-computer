// LLAMACPP-PLAN v2: the llama.cpp side of the split. Runs the cut-out Qwen half (a GGUF of Qwenputer's 32 widened
// Qwen layers) on rows fed directly as input embeddings, over a binary pipe (stdin -> stdout, little-endian):
//   request:  u32 type, u32 n   then for type ROWS: i32 pos[n] (text positions), f32 rows[n * n_embd]
//             type RESET (1): clear the attention cache, the recurrent state and nothing else (n = 0)
//             type ROWS  (2): decode n rows at the given positions (a prefill or one step); logits for the last row
//             type END   (3): exit 0
//             type TOKENS(4): like ROWS but i32 ids[n] instead of rows (the plain-model control)
//   reply:    i32 status (0 ok), then for ROWS: i32 ids[5], f32 vals[5] (top 5 over the whole vocabulary, which is
//             the text vocabulary in this GGUF). Any non-finite logit is an error, never a reply. status != 0: u32 len, char msg[len], then the helper exits 1.
// Positions follow llama.cpp's M-RoPE text rule [p, p, p, 0], which matches Qwenputer's p4 (three rope sections = p,
// position ids = p). Rows must be finite. One process = one model; it never takes the V100 lock (its parent holds it).
//
// usage: qwen_half_helper MODEL.gguf [n_ctx]
#include "llama.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void put(const void * p, size_t n) {
    if (fwrite(p, 1, n, stdout) != n) { exit(2); }
}

static bool get(void * p, size_t n) { return fread(p, 1, n, stdin) == n; }

[[noreturn]] static void fail(const std::string & msg) {
    int32_t st = 1;
    uint32_t len = (uint32_t) msg.size();
    put(&st, 4); put(&len, 4); put(msg.data(), len);
    fflush(stdout);
    fprintf(stderr, "qwen_half_helper: %s\n", msg.c_str());
    exit(1);
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL.gguf [n_ctx]\n", argv[0]); return 1; }
    const int n_ctx = argc > 2 ? atoi(argv[2]) : 16384;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fail("model load failed"); }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = 2048;
    cp.n_ubatch = 2048;
    cp.n_seq_max = 1;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fail("context init failed"); }
    const int n_embd = llama_model_n_embd(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    fprintf(stderr, "qwen_half_helper: ready, n_embd %d, n_vocab %d, n_ctx %d\n", n_embd, n_vocab, n_ctx);
    const int NPOS = 4;                                   // GGML_MROPE_SECTIONS

    std::vector<int32_t> pos_in;
    std::vector<float> rows;
    std::vector<int32_t> toks;
    std::vector<llama_pos> pos4;
    std::vector<int32_t> nseq;
    std::vector<llama_seq_id> seq0;
    std::vector<llama_seq_id *> seqp;
    std::vector<int8_t> outs;
    for (;;) {
        uint32_t hdr[2];
        if (!get(hdr, 8)) { fail("end of input before END"); }
        const uint32_t type = hdr[0], n = hdr[1];
        if (type == 3) { break; }
        if (type == 1) {
            if (n != 0) { fail("RESET with n != 0"); }
            llama_memory_clear(llama_get_memory(ctx), true);
            int32_t st = 0;
            put(&st, 4); fflush(stdout);
            continue;
        }
        if ((type != 2 && type != 4) || n == 0 || n > (uint32_t) n_ctx) { fail("bad request type " + std::to_string(type) + " n " + std::to_string(n)); }
        const bool tok = type == 4;
        pos_in.resize(n);
        if (!get(pos_in.data(), 4 * (size_t) n)) { fail("short message"); }
        if (tok) {
            toks.resize(n);
            if (!get(toks.data(), 4 * (size_t) n)) { fail("short TOKENS message"); }
            for (int32_t t : toks) { if (t < 0 || t >= n_vocab) { fail("token id out of range"); } }
        } else {
            rows.resize((size_t) n * n_embd);
            if (!get(rows.data(), 4 * rows.size())) { fail("short ROWS message"); }
            for (float v : rows) { if (!std::isfinite(v)) { fail("non-finite input row value"); } }
        }
        for (uint32_t i = 0; i < n; ++i) { if (pos_in[i] < 0) { fail("negative position"); } }
        // decode in chunks of n_batch; logits only for the very last row
        const uint32_t nb = cp.n_batch;
        for (uint32_t s = 0; s < n; s += nb) {
            const uint32_t m = std::min(nb, n - s);
            if (tok) {
                pos4.assign(pos_in.begin() + s, pos_in.begin() + s + m);           // llama.cpp expands tokens itself
            } else {
                pos4.assign((size_t) NPOS * m, 0);
                for (uint32_t i = 0; i < m; ++i) {
                    for (int j = 0; j < 3; ++j) { pos4[(size_t) j * m + i] = pos_in[s + i]; }   // [p, p, p, 0]
                }
            }
            nseq.assign(m, 1);
            seq0.assign(m, 0);
            seqp.resize(m);
            for (uint32_t i = 0; i < m; ++i) { seqp[i] = &seq0[i]; }
            outs.assign(m, 0);
            if (s + m == n) { outs[m - 1] = 1; }
            llama_batch b = {};
            b.n_tokens = (int32_t) m;
            b.token = tok ? toks.data() + s : nullptr;
            b.embd = tok ? nullptr : rows.data() + (size_t) s * n_embd;
            b.pos = pos4.data();
            b.n_seq_id = nseq.data();
            b.seq_id = seqp.data();
            b.logits = outs.data();
            const int rc = llama_decode(ctx, b);
            if (rc != 0) { fail("llama_decode returned " + std::to_string(rc)); }
        }
        const float * lg = llama_get_logits_ith(ctx, -1);
        if (!lg) { fail("no logits"); }
        int32_t ids[5];
        float vals[5];
        for (int k = 0; k < 5; ++k) { ids[k] = -1; vals[k] = -INFINITY; }
        for (int t = 0; t < n_vocab; ++t) {
            const float v = lg[t];
            if (!std::isfinite(v)) { fail("non-finite logit at id " + std::to_string(t)); }
            if (v > vals[4]) {
                int k = 4;
                while (k > 0 && v > vals[k - 1]) { vals[k] = vals[k - 1]; ids[k] = ids[k - 1]; --k; }
                vals[k] = v; ids[k] = t;
            }
        }
        int32_t st = 0;
        put(&st, 4); put(ids, sizeof ids); put(vals, sizeof vals);
        fflush(stdout);
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
