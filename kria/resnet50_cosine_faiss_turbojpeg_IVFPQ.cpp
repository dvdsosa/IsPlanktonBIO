/*
 * resnet50_cosine_faiss_turbo.cpp
 *
 * Hybrid pipeline combining:
 *   - Native FAISS search (IndexFlatIP or auto-converted IndexIVFPQ) for fast
 *     approximate cosine-similarity nearest-neighbour classification.
 *   - TurboJPEG image decoding: reads raw JPEG bytes directly from RAM/disk
 *     and decompresses via libturbojpeg, skipping OpenCV's JPEG decoder.
 *   - 4-stage pipeline:
 *       Thread 1 (Producer)    – scans folder, reads + TurboJPEG-decodes images.
 *       Thread 2 (Consumer)    – crops / normalises images for DPU input.
 *       Thread 3 (DPU)         – runs inference, extracts + L2-normalises features.
 *       Thread 4 (FAISS)       – nearest-neighbour search in FAISS index.
 *   - Chrome Tracing output (trace.json): open in chrome://tracing or
 *     Perfetto UI (https://ui.perfetto.dev) to visualise pipeline bottlenecks.
 *   - On first run with an IndexFlatIP/L2 the program automatically converts it
 *     to IndexIVFPQ, saves the cache to disk, and uses it on every subsequent run.
 *
 * IMPORTANT: Requires a backbone-only xmodel that outputs a 2048-dim feature
 * vector (ResNet50 encoder without the linear classification head).
 *
 * Usage:
 *   ./resnet50_cosine_faiss_turbo <backbone.xmodel> <folder_path> \
 *       <faiss_index.bin> <labels.sqlite> [limit] [nprobe]
 *
 * cp -r ../DYB-linearHead/test /dev/shm/test_dataset_ram
 * export LD_LIBRARY_PATH=/home/petalinux/zvec/build/lib:$LD_LIBRARY_PATH
 * 
 * ./bin/resnet50_cosine_faiss_turbojpeg_IVFPQ --config configs/default_config_paper2.yaml
 * 
 * Example:
 *   # copy dataset to RAM first for maximum throughput
 *   cp -r ../DYB-linearHead/test /dev/shm/test_dataset_ram
 *   export LD_LIBRARY_PATH=/usr/local/lib64:/usr/local/lib:$LD_LIBRARY_PATH
 *   ./resnet50_cosine_faiss_turbo ofa_backbone_stg1.xmodel \
 *       /dev/shm/test_dataset_ram data/st1_p089.bin data/st1_p089.sqlite
 *
 * Build (handled by the VS Code task, but here for reference):
 *   g++ -std=c++17 -Ofast -I. -I/usr/local/include -I/usr/include/opencv4 \
 *       -I/usr/include/python3.10 \
 *       -o resnet50_cosine_faiss_turbo resnet50_cosine_faiss_turbo.cpp \
 *       -L/usr/local/lib64 -L/usr/local/lib \
 *       -lfaiss -lturbojpeg -lopenblas -lpython3.10 \
 *       -lvart-runner -lvart-util -lxir -lsqlite3 \
 *       -pthread -lglog -ljson-c \
 *       -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui
 *
 * Copyright 2019 Xilinx Inc. (original pipeline scaffolding).
 * Modified 2026 – FAISS integration, TurboJPEG decoding, Chrome Tracing,
 * 4-stage pipeline.
 */

// ── Standard & system headers ─────────────────────────────────────────────
#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <typeinfo>
#include <variant>
#include <vector>
#include <unordered_map>

// ── Image processing ───────────────────────────────────────────────────────
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <turbojpeg.h>

// ── Vitis-AI / VART ───────────────────────────────────────────────────────
#include <xir/graph/graph.hpp>
#include "vart/runner.hpp"
#include "vart/runner_ext.hpp"
#include "vart/tensor_buffer.hpp"
#include "vitis/ai/collection_helper.hpp"

// ── Python (metric computation) ───────────────────────────────────────────
#include <Python.h>

// ── SQLite (label lookup) ─────────────────────────────────────────────────
#include <sqlite3.h>

// ── Native FAISS ──────────────────────────────────────────────────────────
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/index_io.h>

// Note: yaml-cpp headers are not available in this environment.
// We'll implement a minimal YAML reader tailored to our keys.
// ── JSON (batch flush optional) ────────────────────────────────────────────
#include <json-c/json.h>

namespace fs = std::filesystem;

static std::mutex g_dpu_mutex;

// ============================================================================
// Global config & normalization from YAML
// ============================================================================

struct PipelineConfig {
    // Paths
    std::string dataset_root_path;
    std::string faiss_index_stage_1;
    std::string faiss_index_stage_2;
    std::string sqlite_db_stage_1;
    std::string sqlite_db_stage_2;
    std::string ckpt_stage_1;
    std::string ckpt_stage_2;
    std::string areas_json;
    std::string species_json;
    std::string lehette_params;

    // Transforms
    float s1_mean[3]  = {0.0418f, 0.0353f, 0.0409f};
    float s1_std[3]   = {0.0956f, 0.0911f, 0.0769f};
    float s2_mean[3]  = {0.0425f, 0.0436f, 0.0432f};
    float s2_std[3]   = {0.1466f, 0.1488f, 0.1476f};

    // Numeric params
    double pixel_to_mm2 = 0.0000544644; // default

    // Cosine-similarity OOD thresholds (mirror engine.py). A sample is rejected
    // as Out-Of-Distribution when its nearest-neighbour similarity is *below* the
    // threshold. Defaults disable the gate (no sample is ever rejected); set them
    // in the YAML `runtime:` section to reproduce the Python `ood_sample_*` labels.
    double cosine_threshold_stage_1 = -1.0e30;
    double cosine_threshold_stage_2 = -1.0e30;

    // Runtime
    int  json_flush_interval = 200;
    int  q12_max = 40;
    bool stage2_enabled = true;
    bool trace_enabled  = true;
    bool show_progress_bar = false; // if true, show progress bar during loading
    bool compute_metrics   = false; // if true, run compute_metrics() at end
    // When false (default) FAISS searches the exact index as-is, mirroring the
    // Python engine (faiss.read_index + search). Set true to auto-convert flat
    // indices to IVFPQ for speed — but PQ is lossy and lowers the reported cosine
    // similarity, which breaks the OOD thresholds (especially on small Stage-2
    // indices). Keep false for parity with the Python results.
    bool faiss_ivfpq = false;
};

static float g_meanBGR[3]  = {10.455f, 9.0525f, 10.6845f};
static float g_scaleBGR[3] = {0.05086340632f, 0.0429525589f, 0.04089226932f};
static float g2_meanBGR[3]  = {10.455f, 9.0525f, 10.6845f};
static float g2_scaleBGR[3] = {0.05086340632f, 0.0429525589f, 0.04089226932f};

static inline std::string trim(const std::string& s) {
    const char* ws = " \t\r\n";
    size_t b = s.find_first_not_of(ws); if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(ws);  return s.substr(b, e - b + 1);
}
static inline bool starts_with(const std::string& s, const std::string& p) {
    return s.rfind(p, 0) == 0;
}
static inline void parse_float_array(const std::string& line, float out[3]) {
    auto l = line.find('['); auto r = line.find(']');
    if (l == std::string::npos || r == std::string::npos || r <= l) return;
    std::string inside = line.substr(l+1, r-l-1);
    std::replace(inside.begin(), inside.end(), ',', ' ');
    std::istringstream iss(inside);
    for (int i=0;i<3 && iss; ++i) iss >> out[i];
}
static PipelineConfig load_config_from_yaml(const std::string& path) {
    PipelineConfig cfg;
    std::ifstream ifs(path);
    if (!ifs) {
        std::cerr << "Cannot open config: " << path << ". Using defaults and positional args." << std::endl;
        return cfg;
    }
    std::string line; std::string section; std::string subsection;
    while (std::getline(ifs, line)) {
        std::string t = trim(line);
        if (t.empty() || t[0] == '#') continue;
        if (t == "paths:") { section = "paths"; continue; }
        if (t == "transforms:") { section = "transforms"; continue; }
        if (t == "runtime:") { section = "runtime"; continue; }
        if (t.find(":") != std::string::npos) {
            // Key: value or subsection:
            auto pos = t.find(":");
            std::string key = trim(t.substr(0, pos));
            std::string val = trim(t.substr(pos+1));
            if (val.empty()) { // likely a subsection opener
                subsection = key; continue;
            }
            if (section == "paths") {
                auto unq = [](const std::string& s){ if (s.size()>=2 && (s.front()=='\''||s.front()=='"') && s.back()==s.front()) return s.substr(1, s.size()-2); return s; };
                if (key == "dataset_root_path") cfg.dataset_root_path = unq(val);
                else if (key == "faiss_index_stage_1") cfg.faiss_index_stage_1 = unq(val);
                else if (key == "faiss_index_stage_2") cfg.faiss_index_stage_2 = unq(val);
                else if (key == "sqlite_db_stage_1")   cfg.sqlite_db_stage_1   = unq(val);
                else if (key == "sqlite_db_stage_2")   cfg.sqlite_db_stage_2   = unq(val);
                else if (key == "ckpt_stage_1")        cfg.ckpt_stage_1        = unq(val);
                else if (key == "ckpt_stage_2")        cfg.ckpt_stage_2        = unq(val);
                else if (key == "areas_json")          cfg.areas_json          = unq(val);
                else if (key == "species_json")        cfg.species_json        = unq(val);
                else if (key == "lehette_params")      cfg.lehette_params      = unq(val);
            } else if (section == "transforms") {
                if (subsection == "stage1") {
                    if (key == "mean") parse_float_array(t, cfg.s1_mean);
                    else if (key == "std") parse_float_array(t, cfg.s1_std);
                } else if (subsection == "stage2") {
                    if (key == "mean") parse_float_array(t, cfg.s2_mean);
                    else if (key == "std") parse_float_array(t, cfg.s2_std);
                }
            } else if (section == "runtime") {
                if (key == "json_flush_interval") cfg.json_flush_interval = std::stoi(val);
                else if (key == "pixel_to_mm2") cfg.pixel_to_mm2 = std::stod(val);
                else if (key == "cosine_threshold_stage_1") cfg.cosine_threshold_stage_1 = std::stod(val);
                else if (key == "cosine_threshold_stage_2") cfg.cosine_threshold_stage_2 = std::stod(val);
                else if (key == "faiss_ivfpq") {
                    auto v = val; if (!v.empty() && (v.front()=='"' || v.front()=='\'')) v = v.substr(1, v.size()-2);
                    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c){ return std::tolower(c); });
                    cfg.faiss_ivfpq = (v=="1" || v=="true" || v=="yes");
                }
                else if (key == "stage2_enabled") {
                    auto v = val;
                    if (!v.empty() && (v.front()=='"' || v.front()=='\'')) v = v.substr(1, v.size()-2);
                    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c){ return std::tolower(c); });
                    cfg.stage2_enabled = (v=="1" || v=="true" || v=="yes");
                } else if (key == "trace_enabled") {
                    auto v = val; if (!v.empty() && (v.front()=='"' || v.front()=='\'')) v = v.substr(1, v.size()-2);
                    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c){ return std::tolower(c); });
                    cfg.trace_enabled = (v=="1" || v=="true" || v=="yes");
                } else if (key == "show_progress_bar") {
                    auto v = val; if (!v.empty() && (v.front()=='"' || v.front()=='\'')) v = v.substr(1, v.size()-2);
                    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c){ return std::tolower(c); });
                    cfg.show_progress_bar = (v=="1" || v=="true" || v=="yes");
                } else if (key == "compute_metrics") {
                    auto v = val; if (!v.empty() && (v.front()=='"' || v.front()=='\'')) v = v.substr(1, v.size()-2);
                    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c){ return std::tolower(c); });
                    cfg.compute_metrics = (v=="1" || v=="true" || v=="yes");
                }
            }
        } else if (section.empty() && starts_with(t, "pixel_to_mm2:")) {
            auto pos = t.find(":"); cfg.pixel_to_mm2 = std::stod(trim(t.substr(pos+1)));
        } else if (section == "runtime" && starts_with(t, "queue_limits:")) {
            subsection = "queue_limits"; continue;
        } else if (section == "runtime" && subsection == "queue_limits") {
            auto pos = t.find(":"); if (pos!=std::string::npos) {
                std::string key = trim(t.substr(0,pos)); std::string val = trim(t.substr(pos+1));
                if (key == "q12_max") cfg.q12_max = std::stoi(val);
            }
        }
    }
    // Set global Stage-1/2 BGR mean/std from YAML (already parsed values)
    for (int c=0;c<3;++c) {
        g_meanBGR[c]  = cfg.s1_mean[c];
        g_scaleBGR[c] = cfg.s1_std[c];
        g2_meanBGR[c] = cfg.s2_mean[c];
        g2_scaleBGR[c]= cfg.s2_std[c];
    }
    return cfg;
}

// ============================================================================
// IVFPQ tuning constants
// ============================================================================

/// Number of Voronoi cells for the IVF coarse quantizer.
/// sqrt(18 000) ≈ 134; 64 is conservative but trains faster.
static constexpr int IVFPQ_NLIST    = 64;

/// Number of PQ sub-quantizers.  2048 dims / 64 == 32 dims each → 64 B/vector.
static constexpr int IVFPQ_M        = 64;

/// Bits per sub-quantizer (standard = 8 → 256 centroids per sub-space).
static constexpr int IVFPQ_NBITS    = 8;

/// Default number of Voronoi cells probed at search time.
/// Higher → more accurate but slower.  Override with 6th CLI arg.
static constexpr int DEFAULT_NPROBE = 16;

// ============================================================================
// FAISS database + SQLite label store
// ============================================================================

struct FaissDatabase {
    faiss::Index*            index     = nullptr;   ///< Owned; freed in destructor.
    int                      ntotal    = 0;
    int                      dimension = 0;
    int                      nprobe    = DEFAULT_NPROBE;
    std::vector<std::string> labels;                ///< labels[faiss_id] → class name.

    ~FaissDatabase() { delete index; }
};

/// Derive the IVFPQ cache file path from the raw index path.
/// e.g. "data/st1_p089.bin" → "data/st1_p089.ivfpq.bin"
static std::string ivfpq_cache_path(const std::string& index_path) {
    auto dot = index_path.rfind('.');
    if (dot != std::string::npos)
        return index_path.substr(0, dot) + ".ivfpq" + index_path.substr(dot);
    return index_path + ".ivfpq";
}

/**
 * @brief Convert an IndexFlatIP/L2 to IndexIVFPQ.
 *
 * Takes OWNERSHIP of flat_index (sets the pointer to nullptr after deletion)
 * to release the ~143 MB flat buffer before the heavier IVFPQ training.
 *
 * @return Pointer to the new IndexIVFPQ, or nullptr on failure.
 */
static faiss::Index* convertToIVFPQ(faiss::Index*& flat_index) {
    const int   d       = flat_index->d;
    const int   ntotal  = flat_index->ntotal;
    const auto  metric  = flat_index->metric_type;

    std::cout << "Converting IndexFlat → IndexIVFPQ  (nlist=" << IVFPQ_NLIST
              << ", m=" << IVFPQ_M << ", nbits=" << IVFPQ_NBITS << ")..." << std::endl;

    // 1. Reconstruct all vectors from the flat index.
    std::vector<float> all_vectors(static_cast<size_t>(ntotal) * d);
    for (int i = 0; i < ntotal; ++i)
        flat_index->reconstruct(i, all_vectors.data() + static_cast<size_t>(i) * d);

    // 2. Free the flat index NOW to reclaim RAM before heavy training.
    delete flat_index;
    flat_index = nullptr;
    std::cout << "  Flat index freed (saved ~"
              << (static_cast<size_t>(ntotal) * d * 4) / (1024 * 1024)
              << " MB RAM)." << std::endl;

    // 3. Build coarse quantizer + IVFPQ index.
    faiss::IndexIVFPQ* ivfpq = nullptr;
    if (metric == faiss::METRIC_L2) {
        auto* q = new faiss::IndexFlatL2(d);
        ivfpq   = new faiss::IndexIVFPQ(q, d, IVFPQ_NLIST, IVFPQ_M, IVFPQ_NBITS,
                                        faiss::METRIC_L2);
    } else {
        auto* q = new faiss::IndexFlatIP(d);
        ivfpq   = new faiss::IndexIVFPQ(q, d, IVFPQ_NLIST, IVFPQ_M, IVFPQ_NBITS,
                                        faiss::METRIC_INNER_PRODUCT);
    }
    // ivfpq takes ownership of the coarse quantizer.
    ivfpq->own_fields = true;

    // 4. Train the PQ codebooks.
    std::cout << "  Training IVFPQ on " << ntotal << " vectors..." << std::flush;
    auto t0 = std::chrono::high_resolution_clock::now();
    ivfpq->train(ntotal, all_vectors.data());
    double secs = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();
    std::cout << " done (" << std::fixed << std::setprecision(1) << secs << "s)" << std::endl;

    // 5. Add vectors in small batches to keep peak RAM low during insertion.
    constexpr int ADD_BATCH = 2000;
    std::cout << "  Adding " << ntotal << " vectors (batch=" << ADD_BATCH << ")..." << std::flush;
    for (int offset = 0; offset < ntotal; offset += ADD_BATCH) {
        int count = std::min(ADD_BATCH, ntotal - offset);
        ivfpq->add(count, all_vectors.data() + static_cast<size_t>(offset) * d);
    }
    std::cout << " done (" << ivfpq->ntotal << " vectors)" << std::endl;

    // 6. Release the reconstruction buffer.
    std::vector<float>().swap(all_vectors);
    std::cout << "  Reconstruction buffer freed." << std::endl;

    return ivfpq;
}

/**
 * @brief Load a FAISS index and the associated SQLite label mapping.
 *
 * If the loaded index is a flat (exact) index it is automatically converted to
 * IndexIVFPQ and cached to disk so subsequent runs start instantly.
 *
 * @param index_path  Path to the raw index file (FlatIP or existing IVFPQ).
 * @param db_path     Path to the SQLite database containing feature_mappings.
 * @param fdb         Output: populated FaissDatabase.
 * @return true on success, false on failure.
 */
static bool loadFaissDatabase(const std::string& index_path,
                              const std::string& db_path,
                              FaissDatabase&     fdb,
                              bool               use_ivfpq = false) {
    const std::string cache_path    = ivfpq_cache_path(index_path);
    bool              from_cache    = false;

    // ── Exact search path (Python parity): load the index as-is, no PQ. ─────
    // The Python engine searches the original FlatIP index, so the reported
    // inner product equals the true cosine similarity. IVFPQ's product
    // quantization lowers that value and returns approximate neighbours, which
    // collapses Stage-2 confidence below the OOD threshold. Skip it by default.
    if (!use_ivfpq) {
        std::cout << "Loading FAISS index (exact, no IVFPQ): " << index_path << std::endl;
        try {
            fdb.index  = faiss::read_index(index_path.c_str());
            from_cache = true;   // bypass the IVFPQ cache/convert blocks below
        } catch (const std::exception& e) {
            std::cerr << "Failed to load FAISS index: " << e.what() << std::endl;
            return false;
        }
    }

    // ── Try cached IVFPQ first ─────────────────────────────────────────────
    if (!from_cache && fs::exists(cache_path)) {
        std::cout << "Loading cached IVFPQ index: " << cache_path << std::endl;
        try {
            fdb.index  = faiss::read_index(cache_path.c_str());
            from_cache = true;
        } catch (const std::exception& e) {
            std::cerr << "Warning: cached index invalid, rebuilding ("
                      << e.what() << ")" << std::endl;
        }
    }

    // ── Load original flat index if no valid cache ─────────────────────────
    if (!from_cache) {
        std::cout << "Loading FAISS index: " << index_path << std::endl;
        try {
            fdb.index = faiss::read_index(index_path.c_str());
        } catch (const std::exception& e) {
            std::cerr << "Failed to load FAISS index: " << e.what() << std::endl;
            return false;
        }

        // Detect flat index and convert to IVFPQ.
        const bool is_flat = (dynamic_cast<faiss::IndexFlatIP*>(fdb.index) != nullptr ||
                              dynamic_cast<faiss::IndexFlatL2*>(fdb.index) != nullptr ||
                              dynamic_cast<faiss::IndexFlat*>(fdb.index)   != nullptr);
        if (is_flat) {
            std::cout << "Detected IndexFlat — converting to IVFPQ for faster search..."
                      << std::endl;
            // convertToIVFPQ takes ownership and frees fdb.index.
            faiss::Index* ivfpq = convertToIVFPQ(fdb.index);
            if (!ivfpq) {
                std::cerr << "Failed to convert to IVFPQ." << std::endl;
                return false;
            }
            fdb.index = ivfpq;

            // Persist the IVFPQ index for future runs.
            std::cout << "  Saving IVFPQ cache: " << cache_path << std::endl;
            try {
                faiss::write_index(fdb.index, cache_path.c_str());
            } catch (const std::exception& e) {
                std::cerr << "Warning: could not save cache: " << e.what() << std::endl;
            }
        }
    }

    fdb.ntotal    = static_cast<int>(fdb.index->ntotal);
    fdb.dimension = static_cast<int>(fdb.index->d);

    // Apply nprobe if this is an IVFPQ index.
    auto* ivfpq = dynamic_cast<faiss::IndexIVFPQ*>(fdb.index);
    if (ivfpq) {
        ivfpq->nprobe = fdb.nprobe;
        std::cout << "FAISS IVFPQ: " << fdb.ntotal << " vecs × " << fdb.dimension
                  << " dims, nlist=" << ivfpq->nlist
                  << ", m="          << ivfpq->pq.M
                  << ", nprobe="     << ivfpq->nprobe << std::endl;

        // Print compression ratio for user awareness.
        const size_t flat_bytes = static_cast<size_t>(fdb.ntotal) * fdb.dimension * sizeof(float);
        const size_t pq_bytes   = static_cast<size_t>(fdb.ntotal) * ivfpq->pq.M;
        std::cout << "  Compression: " << (flat_bytes / (1024 * 1024)) << " MB → "
                  << (pq_bytes / 1024) << " KB  ("
                  << std::fixed << std::setprecision(0)
                  << static_cast<float>(flat_bytes) / pq_bytes << "×)" << std::endl;
    } else {
        std::cout << "FAISS DB: " << fdb.ntotal << " vecs × " << fdb.dimension
                  << " dims  (metric="
                  << (fdb.index->metric_type == faiss::METRIC_INNER_PRODUCT ? "IP" : "L2")
                  << ")" << std::endl;
    }

    // ── Load labels from SQLite ────────────────────────────────────────────
    sqlite3* db = nullptr;
    if (sqlite3_open_v2(db_path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
        std::cerr << "Cannot open SQLite database: " << db_path
                  << " (" << sqlite3_errmsg(db) << ")" << std::endl;
        return false;
    }
    fdb.labels.resize(fdb.ntotal);
    sqlite3_stmt* stmt   = nullptr;
    const char*   sql    = "SELECT faiss_id, label FROM feature_mappings ORDER BY faiss_id";
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        std::cerr << "SQL error: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        return false;
    }
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const int   faiss_id = sqlite3_column_int(stmt, 0);
        const char* label    = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        if (faiss_id >= 0 && faiss_id < fdb.ntotal && label)
            fdb.labels[faiss_id] = label;
    }
    sqlite3_finalize(stmt);
    sqlite3_close(db);

    std::cout << "FAISS database loaded successfully." << std::endl;
    return true;
}

/// L2-normalise a float vector in-place (equivalent to torch.nn.functional.normalize).
static void l2_normalize(std::vector<float>& vec) {
    float norm_sq = 0.0f;
    for (float v : vec) norm_sq += v * v;
    float norm = std::sqrt(norm_sq);
    if (norm > 1e-8f) {
        const float inv = 1.0f / norm;
        for (float& v : vec) v *= inv;
    }
}

// ============================================================================
// Global result accumulators (written by Thread 4, read after join)
// ============================================================================

std::vector<int> inferred_classes;
std::vector<int> ground_truth_classes;

// ============================================================================
// Chrome Tracing / Perfetto profiler
// ============================================================================

/// Global epoch for relative timestamps (set once in main before threads start).
static std::chrono::high_resolution_clock::time_point g_epoch;

/// Plain-old-data record for a single completed trace event.
struct TraceEvent {
    std::string name;
    std::string thread;
    long long   ts_us;    ///< Start time in µs relative to g_epoch.
    long long   dur_us;   ///< Duration in µs.
};

/**
 * @brief Thread-safe singleton profiler that records Chrome Trace events.
 *
 * Usage:
 *   ScopedTimer _t("event_name", "thread_name", myAccumMs);
 *
 * Dump at program end:
 *   Profiler::instance().dump("trace.json");
 *
 * Then open trace.json in chrome://tracing or https://ui.perfetto.dev.
 */
class Profiler {
public:
    static Profiler& instance() {
        static Profiler p;
        return p;
    }

    void logEvent(const std::string& name, const std::string& threadName,
                  long long start_us, long long duration_us) {
        std::lock_guard<std::mutex> lk(mtx_);
        events_.push_back({name, threadName, start_us, duration_us});
    }

    void dump(const std::string& path) {
        std::lock_guard<std::mutex> lk(mtx_);
        // Build a deterministic thread-id → integer mapping.
        std::map<std::string, int> tidMap;
        int nextTid = 1;
        for (auto& e : events_)
            if (tidMap.find(e.thread) == tidMap.end())
                tidMap[e.thread] = nextTid++;

        std::ofstream ofs(path);
        if (!ofs) { std::cerr << "[Profiler] Cannot write " << path << std::endl; return; }
        ofs << "[\n";
        for (size_t i = 0; i < events_.size(); ++i) {
            auto& e = events_[i];
            ofs << "  {\"name\":\""  << e.name
                << "\",\"cat\":\"pipeline\""
                << ",\"ph\":\"X\""
                << ",\"ts\":"        << e.ts_us
                << ",\"dur\":"       << e.dur_us
                << ",\"pid\":1"
                << ",\"tid\":"       << tidMap[e.thread]
                << ",\"args\":{\"thread\":\"" << e.thread << "\"}}"
                << (i + 1 < events_.size() ? "," : "") << "\n";
        }
        ofs << "]\n";
        ofs.close();
        std::cout << "[Profiler] Trace written to " << path
                  << " (" << events_.size() << " events)" << std::endl;
    }

private:
    Profiler() = default;
    std::mutex              mtx_;
    std::vector<TraceEvent> events_;
};

/**
 * @brief RAII guard that records a Chrome Trace event on destruction.
 *
 * Also accumulates elapsed time (in ms) into the caller-supplied counter for
 * a per-thread average latency summary printed at the end.
 */
class ScopedTimer {
public:
    ScopedTimer(const char* name, const char* thread, double& accumMs)
        : name_(name), thread_(thread), accumMs_(accumMs),
          start_(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        auto end        = std::chrono::high_resolution_clock::now();
        long long start_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                 start_ - g_epoch).count();
        long long dur_us   = std::chrono::duration_cast<std::chrono::microseconds>(
                                 end - start_).count();
        Profiler::instance().logEvent(name_, thread_, start_us, dur_us);
        accumMs_ += dur_us / 1000.0;
    }

private:
    const char* name_;
    const char* thread_;
    double&     accumMs_;
    std::chrono::high_resolution_clock::time_point start_;
};

// ============================================================================
// Synchronisation primitives
// ============================================================================

/**
 * @brief Counting semaphore used to signal pipeline shutdown between stages.
 *
 * Stage N calls notify() when it has finished; Stage N+1 drains its queue
 * and calls try_acquire() to detect the signal.
 */
class Semaphore {
public:
    explicit Semaphore(int count_ = 0) : count(count_) {}

    void notify() {
        std::unique_lock<std::mutex> lk(mutex);
        ++count;
        cv.notify_one();
    }
    void wait() {
        std::unique_lock<std::mutex> lk(mutex);
        while (count == 0) cv.wait(lk);
        --count;
    }
    bool try_acquire() {
        std::unique_lock<std::mutex> lk(mutex);
        if (count > 0) { --count; return true; }
        return false;
    }
    /// Non-consuming peek: true when the counter is positive (i.e. no token is
    /// currently held). Used by the DPU thread to detect that the pipeline is
    /// idle before tearing down, avoiding the Stage-2 loop-back shutdown race.
    bool is_idle() {
        std::unique_lock<std::mutex> lk(mutex);
        return count > 0;
    }

private:
    std::mutex              mutex;
    std::condition_variable cv;
    int                     count;
};

// ============================================================================
// Pipeline data container
// ============================================================================

/**
 * @brief Carries a mini-batch of images through the pipeline stages.
 *
 * The `features` field is populated by Thread 3 (DPU) and consumed by
 * Thread 4 (FAISS search).
 */
class Image {
public:
    std::vector<cv::Mat>     batchImages;       ///< Raw / preprocessed frames.
    std::vector<int>         batchLabels;        ///< Ground-truth class ids.
    std::vector<std::string> batchImagePaths;    ///< Source paths (for debugging).
    std::vector<float>       features;           ///< 2048-dim L2-normalised embedding.
    // Extended metadata for Stage-2
    bool                      seg_ok = false;
    bool                      touch_ok = false;
    double                    mask_area = 0.0;
    cv::Mat                   mask;              ///< Single-channel 0/255 mask
    cv::Mat                   cropped_bgr;       ///< Cropped original by mask
    std::string               label1;            ///< Stage-1 predicted label string
    float                     sim1 = 0.0f;       ///< Stage-1 cosine similarity (carried to Stage-2).
    cv::Mat                   orig_bgr;          ///< Original BGR for deferred segmentation
};

/// Lock-free-style bounded MPSC queue (single-threaded producer + consumer).
template <typename T>
class SafeQueue {
public:
    explicit SafeQueue(size_t cap = 0) : capacity(cap) {}
    void push(T value) {
        std::unique_lock<std::mutex> lk(mutex);
        cond_not_full.wait(lk, [&]{ return capacity==0 || queue.size() < capacity; });
        queue.push(std::move(value));
        cond_not_empty.notify_one();
    }
    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lk(mutex);
        if (queue.empty()) return false;
        value = std::move(queue.front());
        queue.pop();
        cond_not_full.notify_one();
        return true;
    }
    bool empty() const { std::lock_guard<std::mutex> lk(mutex); return queue.empty(); }

private:
    std::queue<T>           queue;
    std::mutex              mutex;
    std::condition_variable cond_not_empty;
    std::condition_variable cond_not_full;
    size_t                  capacity = 0; // 0 = unbounded
};

// Jobs for shared DPU/FAISS threads with separate queues
struct DpuJob {
    bool         is_stage2 = false;
    cv::Mat      input_int8;      // for Stage-1
    cv::Mat      cropped_bgr;     // for Stage-2
    int          gt_label = -1;
    std::string  path;
    bool         seg_ok = false;
    bool         touch_ok = false;
    double       mask_area = 0.0;
    std::string  label1;          // Stage-1 predicted label string
};

struct FaissJob {
    bool                is_stage2 = false;
    std::vector<float>  features;
    int                 gt_label = -1;
    std::string         path;
    bool                seg_ok = false;
    bool                touch_ok = false;
    double              mask_area = 0.0;
    cv::Mat             cropped_bgr;
    std::string         label1;
};

// ============================================================================
// Pre-processing helpers
// ============================================================================

/// Centre-crop `image` to `height × width`.
static cv::Mat croppedImage(const cv::Mat& image, int height, int width) {
    const int offset_h = (image.rows - height) / 2;
    const int offset_w = (image.cols - width)  / 2;
    return image(cv::Rect(offset_w, offset_h, width, height)).clone();
}

/// Resize so the shorter side is 256, then centre-crop to `size`.
static cv::Mat preprocess_image(const cv::Mat& image, cv::Size size) {
    const float smallest_side = 256.f;
    const float scale = smallest_side /
                        static_cast<float>(std::min(image.rows, image.cols));
    cv::Mat resized;
    cv::resize(image, resized,
               cv::Size(static_cast<int>(image.cols * scale),
                        static_cast<int>(image.rows * scale)));
    return croppedImage(resized, size.height, size.width);
}

/**
 * @brief Normalise + int8-quantise a BGR uint8 image, channel-wise.
 *
 * Equivalent to the old per-pixel double loop, but built as a 256-entry
 * per-channel lookup table applied with cv::LUT (NEON-accelerated on the Kria
 * ARM cores).  Quantisation uses round-to-nearest (std::round, half away from
 * zero); the int8 byte layout is unchanged, so the rest of the pipeline
 * (assignImageDpu) is unaffected.
 *
 * Statistics for the plankton dataset (BGR order):
 *   mean  = [10.455, 9.0525, 10.6845]  (= RGB mean × 255)
 *   scale = 1 / (255 × std_BGR)
 */
static cv::Mat normalize_to_int8(const cv::Mat& image, float fix_scale,
                                 const float mean[3], const float scales[3]) {
    cv::Mat lut(1, 256, CV_8UC3);
    cv::Vec3b* p = lut.ptr<cv::Vec3b>(0);
    for (int i = 0; i < 256; ++i) {
        for (int ch = 0; ch < 3; ++ch) {
            float f = ((float)i - mean[ch]) * scales[ch] * fix_scale;
            f = std::max(-128.f, std::min(127.f, f));
            // ROUND (not truncate): truncation biased the ~95%-black background by +1 LSB,
            // collapsing Stage-2 to a single class on the DPU. Must match torchvision rounding.
            p[i][ch] = static_cast<uchar>(static_cast<signed char>(static_cast<int>(std::lround(f))));
        }
    }
    cv::Mat out;
    cv::LUT(image, lut, out);   // per-channel LUT: out keeps image type (CV_8UC3)
    return out;
}

static cv::Mat setImageRGB(const cv::Mat& image, float fix_scale) {
    return normalize_to_int8(image, fix_scale, g_meanBGR, g_scaleBGR);
}

static cv::Mat setImageRGB_Stage2(const cv::Mat& image, float fix_scale) {
    return normalize_to_int8(image, fix_scale, g2_meanBGR, g2_scaleBGR);
}

/// Return 2^fixpoint (input scale from xmodel attribute).
static float get_input_scale(const xir::Tensor* tensor) {
    return std::exp2f((float)tensor->template get_attr<int>("fix_point"));
}

/// Return 2^(-fixpoint) (output scale from xmodel attribute).
static float get_output_scale(const xir::Tensor* tensor) {
    return std::exp2f(-(float)tensor->template get_attr<int>("fix_point"));
}

// ============================================================================
// Forward declaration (definition after pipeline threads)
// ============================================================================
static std::vector<float> convert_fixpoint_to_float(vart::TensorBuffer* tensor,
                                                     float scale, int batch_idx);
// Segmentation + helpers forward declarations
static cv::Mat adjust_brightness_contrast_cv(const cv::Mat& gray, int brightness, int contrast);
static bool is_border_touch_acceptable(const cv::Mat& mask, double tolerance_percent);
static std::tuple<bool,double,cv::Mat,cv::Mat> segment_and_crop(const cv::Mat& bgr);
static inline std::pair<std::string,std::string> get_subtitles(const std::string& path);

/// Build a JSON number rounded to 6 decimals (mirrors Python's round(x, 6) +
/// json.dump minimal representation, e.g. 0.9 → "0.9", 6.621422 → "6.621422").
static json_object* json_rounded_double(double v) {
    double r = std::round(v * 1e6) / 1e6;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6f", r);
    std::string s(buf);
    if (s.find('.') != std::string::npos) {
        size_t last = s.find_last_not_of('0');
        if (s[last] == '.') --last;
        s.erase(last + 1);
    }
    return json_object_new_double_s(r, s.c_str());
}

// Biomass parameters loader
static std::map<std::string, std::pair<double,double>> load_biomass_params(const std::string& json_path) {
    std::map<std::string, std::pair<double,double>> res;
    json_object* root = json_object_from_file(json_path.c_str());
    if (!root) return res;
    json_object_object_foreach(root, key, val) {
        json_object* a = nullptr; json_object* b = nullptr;
        if (json_object_object_get_ex(val, "a", &a) && json_object_object_get_ex(val, "b", &b)) {
            res[key] = {json_object_get_double(a), json_object_get_double(b)};
        }
    }
    json_object_put(root);
    return res;
}

// ============================================================================
// Image loading with TurboJPEG
// ============================================================================

/**
 * @brief Reads JPEG images from disk and decodes them via libturbojpeg.
 *
 * Benefits over cv::imread:
 *   - libturbojpeg uses SIMD-optimised Huffman decoding — typically 2-4× faster.
 *   - When images reside in /dev/shm (RAM filesystem) the read() syscall is a
 *     memcpy, making I/O essentially free.
 *   - A single reusable file-buffer avoids repeated heap allocations.
 */
class ImageLoader {
public:
    ImageLoader(const std::vector<std::string>& imagePaths,
                const std::vector<int>&         labels,
                int                             batchSize,
                cv::Size                        imgSize)
        : imagePaths_(imagePaths), labels_(labels),
          batchSize_(batchSize), imgSize_(imgSize), index_(0)
    {
        tjHandle_ = tjInitDecompress();
        if (!tjHandle_)
            LOG(FATAL) << "TurboJPEG: failed to initialise decompressor.";
    }

    ~ImageLoader() {
        if (tjHandle_) tjDestroy(tjHandle_);
    }

    // Non-copyable — owns the tj handle.
    ImageLoader(const ImageLoader&)            = delete;
    ImageLoader& operator=(const ImageLoader&) = delete;

    bool nextBatch(std::vector<cv::Mat>&        batchImages,
                   std::vector<int>&            batchLabels,
                   std::vector<std::string>&    batchImagePaths) {
        if (index_ >= imagePaths_.size()) return false;
        for (int i = 0; i < batchSize_ && index_ < imagePaths_.size(); ++i) {
            cv::Mat img = decodeTurboJPEG(imagePaths_[index_]);
            if (img.empty()) {
                LOG(WARNING) << "TurboJPEG: skipping unreadable file: " << imagePaths_[index_];
                ++index_;
                continue;
            }
            batchImages.push_back(std::move(img));
            batchLabels.push_back(labels_[index_]);
            batchImagePaths.push_back(imagePaths_[index_]);
            ++index_;
        }
        return true;
    }

private:
    /**
     * @brief Read the JPEG file and decompress directly into a BGR cv::Mat.
     *
     * The file buffer `fileBuf_` is reused across calls to avoid reallocation.
     */
    cv::Mat decodeTurboJPEG(const std::string& path) {
        // 1. Read entire file into memory.
        std::ifstream ifs(path, std::ios::binary | std::ios::ate);
        if (!ifs) return {};
        const auto fileSize = ifs.tellg();
        if (fileSize <= 0) return {};
        ifs.seekg(0);
        fileBuf_.resize(static_cast<size_t>(fileSize));
        ifs.read(reinterpret_cast<char*>(fileBuf_.data()), fileSize);
        if (!ifs) return {};

        // 2. Parse JPEG header to get dimensions.
        int width = 0, height = 0, jpegSubsamp = 0, jpegColorspace = 0;
        if (tjDecompressHeader3(tjHandle_, fileBuf_.data(),
                                static_cast<unsigned long>(fileSize),
                                &width, &height, &jpegSubsamp,
                                &jpegColorspace) != 0) {
            LOG(WARNING) << "TurboJPEG header error: " << tjGetErrorStr();
            return {};
        }

        // 3. Decompress directly into pre-allocated cv::Mat (BGR output).
        cv::Mat img(height, width, CV_8UC3);
        if (tjDecompress2(tjHandle_, fileBuf_.data(),
                          static_cast<unsigned long>(fileSize),
                          img.data, width, /*pitch=*/0, height,
                          TJPF_BGR,
                          TJFLAG_ACCURATEDCT | TJFLAG_NOREALLOC) != 0) {
            LOG(WARNING) << "TurboJPEG decompress error: " << tjGetErrorStr();
            return {};
        }
        return img;
    }

    std::vector<std::string>     imagePaths_;
    std::vector<int>             labels_;
    int                          batchSize_;
    cv::Size                     imgSize_;
    size_t                       index_;
    tjhandle                     tjHandle_  = nullptr;
    std::vector<unsigned char>   fileBuf_;   ///< Reused across calls.
};

// ============================================================================
// Dataset helpers
// ============================================================================

/// Recursively collect all .jpg file paths under `rootFolder` (up to `limit`).
std::vector<std::string> getAllImagePaths(const std::string& rootFolder, int limit) {
    std::vector<std::string> imagePaths;
    const fs::path rootPath(rootFolder);
    if (!fs::exists(rootPath) || !fs::is_directory(rootPath)) {
        std::cerr << "Invalid root folder: " << rootFolder << std::endl;
        return imagePaths;
    }
    for (const auto& entry : fs::recursive_directory_iterator(rootPath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
            imagePaths.push_back(entry.path().string());
            if (limit > 0 && static_cast<int>(imagePaths.size()) >= limit) break;
        }
    }
    return imagePaths;
}

/// Extract the numeric class id from the parent directory name (e.g. "42_ClassName" → 42).
std::vector<int> extractNumbersFromPaths(const std::vector<std::string>& imagePaths) {
    std::vector<int> numbers;
    for (const auto& path : imagePaths) {
        const std::string parentDir = fs::path(path).parent_path().filename().string();
        numbers.push_back(std::stoi(parentDir.substr(0, parentDir.find('_'))));
    }
    return numbers;
}

/// Print a simple ASCII progress bar to stdout (overwriting the current line).
void printProgress(double percentage) {
    constexpr int barWidth = 70;
    std::cout << "[";
    const int pos = static_cast<int>(barWidth * percentage);
    for (int i = 0; i < barWidth; ++i)
        std::cout << (i < pos ? '=' : i == pos ? '>' : ' ');
    std::cout << "] " << static_cast<int>(percentage * 100.0) << " %\r";
    std::cout.flush();
}

// ============================================================================
// Thread 1 – Producer: scan folder & decode images via TurboJPEG
// ============================================================================

/**
 * @brief Reads all images from `rootFolder` and pushes them onto `queue`.
 *
 * Signals `sem` after the last image so Thread 2 can detect EOF.
 * Each read + TurboJPEG decode is timed and reported via ScopedTimer.
 */
void imageProducer(SafeQueue<Image>& queue, Semaphore& sem,
                   int& limit, std::string& rootFolder,
                   bool show_progress_bar) {
    const int    batch_size = 1;
    auto imagePaths         = getAllImagePaths(rootFolder, limit);
    auto labels             = extractNumbersFromPaths(imagePaths);
    if (limit == 0) limit   = static_cast<int>(labels.size());

    ImageLoader loader(imagePaths, labels, batch_size, cv::Size(224, 224));
    bool   cont     = true;
    int    count    = 0;
    double accumMs  = 0.0;

    while (cont && count < limit) {
        std::vector<cv::Mat>     bI;
        std::vector<int>         bL;
        std::vector<std::string> bP;
        {
            ScopedTimer _t("Read_Decode", "Producer", accumMs);
            cont = loader.nextBatch(bI, bL, bP);
        }
        if (!cont || bI.empty()) break;

        Image img;
        img.batchImages     = std::move(bI);
        img.batchLabels     = std::move(bL);
        img.batchImagePaths = std::move(bP);
        queue.push(std::move(img));
        ++count;
        if (show_progress_bar)
            printProgress(static_cast<double>(count) / limit);
    }

    if (count > 0) {
        if (show_progress_bar) std::cout << "\n";
        std::cout << "[Perf] Read_Decode Avg: " << std::fixed << std::setprecision(2)
                  << (accumMs / count) << " ms/img" << std::endl;
    }
    sem.notify();
}

// ============================================================================
// Thread 2 – Consumer/Preprocessor: resize, crop, normalise
// ============================================================================

/**
 * @brief Pops raw images from `queue`, applies preprocessing, and pushes to `queue2`.
 *
 * Preprocessing:
 *   1. Resize shorter side to 256px.
 *   2. Centre-crop to 224×224.
 *   3. Subtract dataset mean, divide by std, scale to DPU int8 range.
 *
 * Signals `semAC` after draining the queue when `semB` (from Thread 1) is set.
 */
void imageConsumer(SafeQueue<Image>& queue, SafeQueue<Image>& q2dpu_s1,
                   Semaphore& semB, Semaphore& semAC, Semaphore& next_image_sem,
                   int& run_batch) {
    // DPU input fix_scale = 2^3 = 8  (verify against xmodel fix_point attribute)
    constexpr float input_scale = 8.f;
    Image  image;
    int    count   = 0;
    double accumMs = 0.0;

    while (true) {
        if (queue.try_pop(image)) {
            next_image_sem.wait();
            ScopedTimer _t("Preprocess", "Consumer", accumMs);
            std::vector<cv::Mat> images(run_batch);
            std::vector<cv::Mat> new_images(run_batch);
            for (int b = 0; b < run_batch; ++b) {
                images[b]     = preprocess_image(image.batchImages[b], cv::Size(224, 224));
                new_images[b] = setImageRGB(images[b], input_scale);
            }
            Image out;
            out.batchImages     = std::move(new_images);
            out.batchLabels     = image.batchLabels;
            out.batchImagePaths = image.batchImagePaths;
            // keep original BGR for deferred segmentation (Stage-1 gating)
            out.orig_bgr        = image.batchImages[0];
            q2dpu_s1.push(std::move(out));
            ++count;
        } else {
            if (semB.try_acquire()) {
                if (count > 0)
                    std::cout << "[Perf] Preprocess Avg: " << std::fixed << std::setprecision(2)
                              << (accumMs / count) << " ms/img" << std::endl;
                semAC.notify();
                break;
            }
        }
    }
}

/// Copy a BGR cv::Mat into the DPU input buffer as RGB int8 (channel reorder).
static void assignImageDpu(const cv::Mat& image, void* data) {
    signed char* d = reinterpret_cast<signed char*>(data);
    int c = 0;
    for (int row = 0; row < image.rows; ++row)
        for (int col = 0; col < image.cols; ++col) {
            const auto v = image.at<cv::Vec3b>(row, col);
            d[c++] = static_cast<signed char>(v[2]); // R
            d[c++] = static_cast<signed char>(v[1]); // G
            d[c++] = static_cast<signed char>(v[0]); // B
        }
}

// ============================================================================
// Segmentation helpers (Stage-2)
// ============================================================================

static cv::Mat adjust_brightness_contrast_cv(const cv::Mat& gray, int brightness, int contrast) {
    // alpha = contrast/100, beta = (brightness-100)*2.55
    double alpha = static_cast<double>(contrast) / 100.0;
    double beta  = static_cast<double>(brightness - 100) * 2.55;
    cv::Mat out; cv::convertScaleAbs(gray, out, alpha, beta);
    return out;
}

static bool is_border_touch_acceptable(const cv::Mat& mask, double tolerance_percent) {
    if (mask.empty() || mask.channels() != 1) return false;
    int h = mask.rows, w = mask.cols;
    int max_pixels = static_cast<int>((tolerance_percent / 100.0) * w);
    int top = cv::countNonZero(mask.row(0));
    int bottom = cv::countNonZero(mask.row(h-1));
    int left = cv::countNonZero(mask.col(0));
    int right = cv::countNonZero(mask.col(w-1));
    int borders_touched = (top>0) + (bottom>0) + (left>0) + (right>0);
    if (borders_touched == 0) return true;
    if (borders_touched == 1) return (top<=max_pixels && bottom<=max_pixels && left<=max_pixels && right<=max_pixels);
    return false;
}

static std::tuple<bool,double,cv::Mat,cv::Mat> segment_and_crop(const cv::Mat& bgr) {
    if (bgr.empty()) return {false, 0.0, cv::Mat(), cv::Mat()};
    cv::Mat gray; cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat adj = adjust_brightness_contrast_cv(gray, 150, 200);
    cv::Mat otsu;
    cv::threshold(adj, otsu, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat kernel_cross = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5,5));
    cv::Mat kernel_ellipse = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5));
    cv::Mat dil, ero; cv::dilate(otsu, dil, kernel_cross, cv::Point(-1,-1), 1);
    cv::erode(dil, ero, kernel_ellipse, cv::Point(-1,-1), 1);
    cv::Mat binv; cv::threshold(ero, binv, 100, 255, cv::THRESH_BINARY_INV);
    cv::Mat bordered; cv::copyMakeBorder(binv, bordered, 10,10,10,10, cv::BORDER_CONSTANT, cv::Scalar(255));
    std::vector<std::vector<cv::Point>> contours; std::vector<cv::Vec4i> hierarchy;
    cv::findContours(bordered, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
    if (contours.empty()) return {false, 0.0, cv::Mat(), cv::Mat()};
    // remove largest contour (background)
    size_t largest_idx = 0; double largest_area = 0.0;
    for (size_t i=0;i<contours.size();++i) { double a = cv::contourArea(contours[i]); if (a>largest_area) { largest_area=a; largest_idx=i; } }
    if (contours.size() <= 1) return {false, largest_area, cv::Mat(), cv::Mat()};
    std::vector<std::vector<cv::Point>> filtered; filtered.reserve(contours.size()-1);
    for (size_t i=0;i<contours.size();++i) if (i!=largest_idx) filtered.push_back(contours[i]);
    // choose largest as organism, shift -10px
    size_t idx2=0; double ar2=0.0; for (size_t i=0;i<filtered.size();++i) { double a=cv::contourArea(filtered[i]); if (a>ar2){ar2=a; idx2=i;} }
    for (auto& p : filtered[idx2]) { p.x = std::max(0, p.x - 10); p.y = std::max(0, p.y - 10); }
    double mask_area = ar2;
    cv::Mat mask = cv::Mat::zeros(bgr.rows, bgr.cols, CV_8UC1);
    cv::drawContours(mask, std::vector<std::vector<cv::Point>>{filtered[idx2]}, -1, cv::Scalar(255), cv::FILLED);
    cv::Mat cropped; cv::bitwise_and(bgr, bgr, cropped, mask);
    return {true, mask_area, mask, cropped};
}

static inline std::pair<std::string,std::string> get_subtitles(const std::string& path) {
    fs::path p(path);
    return {p.parent_path().filename().string(), p.filename().string()};
}

// ============================================================================
// Alternating MUX versions of DPU and FAISS (single-runner DPU, stage2 offloaded)
// ============================================================================

// DPU MUX: process Stage-2 requests with priority; else Stage-1
static void dpuTaskMux(SafeQueue<Image>& q2dpu_s1,
                       SafeQueue<Image>& q2dpu_s2,
                       Semaphore& semAC,
                       Semaphore& semZ,
                       SafeQueue<Image>& qdpu_s1,
                       SafeQueue<Image>& qdpu_s2,
                       std::unique_ptr<vart::RunnerExt>& runner1,
                       std::unique_ptr<vart::RunnerExt>& runner2,
                       int& run_batch,
                       Semaphore& next_image_sem) {
    auto in_bufs1  = runner1->get_inputs();
    auto out_bufs1 = runner1->get_outputs();
    CHECK_EQ(in_bufs1.size(),  1u);
    CHECK_EQ(out_bufs1.size(), 1u);
    auto out_tensor1 = out_bufs1[0]->get_tensor();
    auto out_scale1  = get_output_scale(out_tensor1);
    std::vector<vart::TensorBuffer*> in_bufs2, out_bufs2; float out_scale2=0.0f;
    if (runner2) {
        in_bufs2  = runner2->get_inputs(); out_bufs2 = runner2->get_outputs();
        CHECK_EQ(in_bufs2.size(),1u); CHECK_EQ(out_bufs2.size(),1u);
        out_scale2 = get_output_scale(out_bufs2[0]->get_tensor());
        // One-time runner2 IO shape trace
        auto ish = in_bufs2[0]->get_tensor()->get_shape();
        auto osh = out_bufs2[0]->get_tensor()->get_shape();
        std::cout << "[Trace] S2 DPU IO shapes in=[";
        for (size_t i=0;i<ish.size();++i) std::cout<<ish[i]<<(i+1<ish.size()?", ":"");
        std::cout << "] out=[";
        for (size_t i=0;i<osh.size();++i) std::cout<<osh[i]<<(i+1<osh.size()?", ":"");
        std::cout << "]\n";
    }
    double accumMs = 0.0; int count=0;
    Image item;
    while (true) {
        bool did=false;
        if (q2dpu_s2.try_pop(item)) {
            ScopedTimer _t("DPU_Inference(Stage2)", "DPU", accumMs);
            // The pipeline runs with a single next_image_sem token (one image in
            // flight at a time). Every early-exit below drops this Stage-2 image,
            // so it MUST return its token — otherwise the Consumer thread blocks
            // forever on next_image_sem.wait() and the whole pipeline hangs.
            if (!runner2) { next_image_sem.notify(); continue; }
            if (item.batchImages.empty() || item.batchImages[0].empty()) { next_image_sem.notify(); continue; }
            if (!item.batchImages[0].isContinuous()) item.batchImages[0] = item.batchImages[0].clone();
            uint64_t data_in=0; size_t size_in=0; std::tie(data_in,size_in)= in_bufs2[0]->data({0,0,0,0});
            if (size_in==0) { next_image_sem.notify(); continue; }
            assignImageDpu(item.batchImages[0], reinterpret_cast<void*>(data_in));
            auto t2 = in_bufs2[0]->get_tensor(); auto sz2 = t2->get_data_size()/std::max<int>(1, t2->get_shape()[0]);
            in_bufs2[0]->sync_for_write(0, sz2);
            {
                std::lock_guard<std::mutex> lk(g_dpu_mutex);
                auto v = runner2->execute_async(in_bufs2, out_bufs2);
                auto st = runner2->wait((int)v.first, -1);
                if (st!=0) { std::cout<<"[Trace] S2 DPU wait failed status="<<st<<"\n"; next_image_sem.notify(); continue; }
            }
            auto to2 = out_bufs2[0]->get_tensor(); auto so2 = to2->get_data_size()/std::max<int>(1, to2->get_shape()[0]);
            out_bufs2[0]->sync_for_read(0, so2);
            auto feat = convert_fixpoint_to_float(out_bufs2[0], out_scale2, 0); l2_normalize(feat);
            item.features = std::move(feat);
            qdpu_s2.push(std::move(item)); did=true; ++count;
        } else if (q2dpu_s1.try_pop(item)) {
            ScopedTimer _t("DPU_Inference(Stage1)", "DPU", accumMs);
            // Same token-return requirement for dropped Stage-1 images (see above).
            if (item.batchImages.empty() || item.batchImages[0].empty()) { next_image_sem.notify(); continue; }
            if (!item.batchImages[0].isContinuous()) item.batchImages[0] = item.batchImages[0].clone();
            uint64_t data_in=0; size_t size_in=0; std::tie(data_in,size_in)= in_bufs1[0]->data({0,0,0,0});
            if (size_in==0) { next_image_sem.notify(); continue; }
            assignImageDpu(item.batchImages[0], reinterpret_cast<void*>(data_in));
            auto t1 = in_bufs1[0]->get_tensor(); auto sz1 = t1->get_data_size()/std::max<int>(1, t1->get_shape()[0]);
            in_bufs1[0]->sync_for_write(0, sz1);
            {
                std::lock_guard<std::mutex> lk(g_dpu_mutex);
                auto v = runner1->execute_async(in_bufs1, out_bufs1); auto st = runner1->wait((int)v.first, -1); CHECK_EQ(st,0);
            }
            auto to1 = out_bufs1[0]->get_tensor(); auto so1 = to1->get_data_size()/std::max<int>(1, to1->get_shape()[0]);
            out_bufs1[0]->sync_for_read(0, so1);
            auto feat = convert_fixpoint_to_float(out_bufs1[0], out_scale1, 0); l2_normalize(feat);
            item.features = std::move(feat);
            qdpu_s1.push(std::move(item)); did=true; ++count;
        } else {
            // Shutdown-race fix: only tear down when the Consumer is done (semAC)
            // AND no image is still in flight (next_image_sem idle). Without the
            // idle check, a Stage-1 image being handled by the FAISS thread could
            // still loop back into q2dpu_s2 for Stage-2 *after* this thread exited,
            // orphaning it (dropped from the results) and leaking its token.
            // is_idle() is a non-consuming peek, so semAC is only taken on a real
            // shutdown; short-circuit keeps semAC untouched while work remains.
            if (next_image_sem.is_idle() && semAC.try_acquire()) {
                if (count>0) std::cout << "[Perf] DPU_Inference Avg: " << std::fixed << std::setprecision(2) << (accumMs/count) << " ms/img\n";
                semZ.notify();
                break;
            }
        }
    }
}

/// One inference record, mirroring engine.py::_accumulate_result.
/// For this cosine-FAISS pipeline the per-stage "confidence" is the cosine
/// similarity (inner product of L2-normalised vectors) returned by FAISS.
struct ResultEntry {
    std::string filename;
    std::string s1_pred;  bool s1_pred_null = true;
    double      s1_conf = 0.0;  bool s1_conf_null = true;
    std::string s2_pred;  bool s2_pred_null = true;
    double      s2_conf = 0.0;  bool s2_conf_null = true;
    std::string status;
    double      biomass = 0.0;  bool biomass_null = true;
};

// FAISS MUX: process Stage-2 results with priority; else Stage-1; coordinate alternation.
//
// Routing mirrors resnet50_cosine_linear_energy_turbojpeg.cpp: ground truth is
// NEVER used to gate the pipeline. Stage-1 always proceeds to segmentation, and a
// sample is "matched" only when the Stage-1 prediction agrees with the Stage-2
// prediction (not against ground truth).
static void faissTaskMux(SafeQueue<Image>& qdpu_s1,
                         SafeQueue<Image>& qdpu_s2,
                         SafeQueue<Image>& q2dpu_s2,
                         Semaphore& semZ,
                         const FaissDatabase& fdb1,
                         const FaissDatabase& fdb2,
                         const PipelineConfig& cfg,
                         const std::map<std::string, std::pair<double,double>>& biomass_params,
                         Semaphore& next_image_sem) {
    constexpr int k=1; float d[k]; faiss::idx_t I[k];
    // areas_json:   class_id -> { image_id : area_mm2 }
    std::map<std::string, std::map<std::string, std::string>> areas_acc;
    // species_json: ground_truth_class -> [ ResultEntry, ... ]  (engine.py results_dict)
    std::map<std::string, std::vector<ResultEntry>> results;

    auto flush=[&](){
        // areas json
        try {
            fs::create_directories(fs::path(cfg.areas_json).parent_path());
            json_object* root=json_object_new_object();
            for(auto&[cls,mp]:areas_acc){ json_object*o=json_object_new_object(); for(auto&[f,a]:mp) json_object_object_add(o,f.c_str(),json_object_new_string(a.c_str())); json_object_object_add(root,cls.c_str(),o);}
            std::ofstream ofs(cfg.areas_json); if(ofs){ ofs<<json_object_to_json_string_ext(root,JSON_C_TO_STRING_PRETTY);} json_object_put(root);
        } catch(...){}
        // species/results json (engine.py structure)
        try {
            fs::create_directories(fs::path(cfg.species_json).parent_path());
            json_object* root=json_object_new_object();
            for(auto& [gt, vec] : results){
                json_object* arr=json_object_new_array();
                for(auto& e : vec){
                    json_object* ent=json_object_new_object();
                    json_object_object_add(ent,"filename", json_object_new_string(e.filename.c_str()));
                    json_object_object_add(ent,"stage_1_prediction", e.s1_pred_null ? json_object_new_null() : json_object_new_string(e.s1_pred.c_str()));
                    json_object_object_add(ent,"stage_1_confidence", e.s1_conf_null ? json_object_new_null() : json_rounded_double(e.s1_conf));
                    json_object_object_add(ent,"stage_2_prediction", e.s2_pred_null ? json_object_new_null() : json_object_new_string(e.s2_pred.c_str()));
                    json_object_object_add(ent,"stage_2_confidence", e.s2_conf_null ? json_object_new_null() : json_rounded_double(e.s2_conf));
                    json_object_object_add(ent,"pipeline_status", json_object_new_string(e.status.c_str()));
                    json_object_object_add(ent,"biomass_mg", e.biomass_null ? json_object_new_null() : json_rounded_double(e.biomass));
                    json_object_array_add(arr, ent);
                }
                json_object_object_add(root, gt.c_str(), arr);
            }
            std::ofstream ofs(cfg.species_json); if(ofs){ ofs<<json_object_to_json_string_ext(root,JSON_C_TO_STRING_PRETTY);} json_object_put(root);
        } catch(...){}
    };
    auto to_int=[](const std::string&s){ try{ return std::stoi(s.substr(0,s.find('_')));}catch(...){return -1;}};
    double accumMs=0.0; int count=0; Image item;
    while(true){
        if (qdpu_s2.try_pop(item)){
            // ── Stage-2: encoder -> FAISS -> (label2, sim2) ──────────────────
            ScopedTimer _t("FAISS_Search(Stage2)", "FAISS", accumMs);
            if (cfg.trace_enabled) std::cout << "[Trace] S2 FAISS start for "<< item.batchImagePaths[0] <<"\n";
            fdb2.index->search(1, item.features.data(), k, d, I);
            std::string label2 = (I[0]>=0 && I[0]<fdb2.ntotal)? fdb2.labels[I[0]]:std::string("unknown");
            float sim2 = d[0];
            if (cfg.trace_enabled) std::cout << "[Trace] S2 FAISS ok label2="<<label2<<"\n";
            auto [cls,file] = get_subtitles(item.batchImagePaths[0]);

            ResultEntry e; e.filename = file; e.status = "unmatched";
            e.s1_pred = item.label1; e.s1_pred_null = item.label1.empty();
            e.s1_conf = item.sim1; e.s1_conf_null = false;
            e.s2_pred = label2; e.s2_pred_null = false;
            e.s2_conf = sim2; e.s2_conf_null = false;

            // Final decision mirrors engine.py: Stage-2 OOD gate, then agreement.
            if (sim2 < cfg.cosine_threshold_stage_2){
                e.status = "ood_sample_stage2";
            } else if (label2 == item.label1){
                e.status = "matched";
                double area_mm2 = item.mask_area * cfg.pixel_to_mm2;
                auto it=biomass_params.find(label2); if(it!=biomass_params.end()){ auto[a,b]=it->second; e.biomass=a*std::pow(area_mm2,b); e.biomass_null=false; }
            } else {
                e.status = "unmatched";
            }
            results[cls].push_back(std::move(e));
            ++count; next_image_sem.notify();
        } else if (qdpu_s1.try_pop(item)){
            // ── Stage-1: encoder -> FAISS -> (label1, sim1) ──────────────────
            float sim1;
            std::string label1;
            {
                ScopedTimer _t("FAISS_Search(Stage1)", "FAISS", accumMs);
                fdb1.index->search(1, item.features.data(), k, d, I);
                label1 = (I[0]>=0 && I[0]<fdb1.ntotal)? fdb1.labels[I[0]]:std::string("unknown");
                sim1 = d[0];
            }
            int pred_id = to_int(label1); int gt_id = item.batchLabels[0];
            auto [cls,file] = get_subtitles(item.batchImagePaths[0]);
            if (cfg.trace_enabled)
                std::cout << "[Trace] S1 label="<<label1<<" gt="<<gt_id<<"\n";

            // Record (predicted, ground-truth) for the optional metric computation only.
            inferred_classes.push_back(pred_id);
            ground_truth_classes.push_back(gt_id);

            // Stage-1 OOD gate (mirrors engine.py). Ground truth is NEVER used as a
            // pipeline gate here — it is only the JSON grouping key — so segmentation
            // and Stage-2 run for every in-distribution sample, exactly like Python.
            if (sim1 < cfg.cosine_threshold_stage_1){
                ResultEntry e; e.filename=file; e.status="ood_sample_stage1";
                e.s1_pred=label1; e.s1_pred_null=false; e.s1_conf=sim1; e.s1_conf_null=false;
                results[cls].push_back(std::move(e));
                next_image_sem.notify();
            } else {
            // Segmentation is required to reach Stage-2 (no ground-truth gating).
            double dummy_accum = 0.0;
            ScopedTimer _t("segment_and_crop", "FAISS", dummy_accum);
            bool seg_ok=false, touch_ok=false; double mask_area=0.0; cv::Mat mask, cropped;
            if (!item.orig_bgr.empty()){
                std::tie(seg_ok, mask_area, mask, cropped) = segment_and_crop(item.orig_bgr);
                if (seg_ok) touch_ok = is_border_touch_acceptable(mask, 10.0);
            }
            if (!seg_ok){
                ResultEntry e; e.filename=file; e.status="failed-segmentation";
                e.s1_pred=label1; e.s1_pred_null=false; e.s1_conf=sim1; e.s1_conf_null=false;
                results[cls].push_back(std::move(e));
                next_image_sem.notify();
            } else if (!touch_ok){
                ResultEntry e; e.filename=file; e.status="touches-border";
                e.s1_pred=label1; e.s1_pred_null=false; e.s1_conf=sim1; e.s1_conf_null=false;
                results[cls].push_back(std::move(e));
                next_image_sem.notify();
            } else {
                // Save area on successful segmentation
                char buf[64]; std::snprintf(buf,sizeof(buf),"%.3f", mask_area * cfg.pixel_to_mm2); areas_acc[cls][file]=std::string(buf);
                if (cropped.empty()){
                    ResultEntry e; e.filename=file; e.status="failed-cropping";
                    e.s1_pred=label1; e.s1_pred_null=false; e.s1_conf=sim1; e.s1_conf_null=false;
                    results[cls].push_back(std::move(e));
                    next_image_sem.notify();
                } else if (!cfg.stage2_enabled) {
                    // Stage-2 disabled: treat the Stage-1 prediction as final.
                    ResultEntry e; e.filename=file; e.status="matched";
                    e.s1_pred=label1; e.s1_pred_null=false; e.s1_conf=sim1; e.s1_conf_null=false;
                    e.s2_pred=label1; e.s2_pred_null=false; e.s2_conf=sim1; e.s2_conf_null=false;
                    double area_mm2 = mask_area * cfg.pixel_to_mm2;
                    auto it=biomass_params.find(label1); if(it!=biomass_params.end()){ auto[a,b]=it->second; e.biomass=a*std::pow(area_mm2,b); e.biomass_null=false; }
                    results[cls].push_back(std::move(e));
                    next_image_sem.notify();
                } else {
                    if (cfg.trace_enabled) std::cout << "[Trace] S2 enqueue "<< file <<"\n";
                    cv::Mat s2 = preprocess_image(cropped, cv::Size(224,224));
                    cv::Mat s2_int8 = setImageRGB_Stage2(s2, 8.f);
                    Image s2item; s2item.batchImages.resize(1);
                    s2item.batchImages[0]=std::move(s2_int8);
                    s2item.batchLabels=item.batchLabels;
                    s2item.batchImagePaths=item.batchImagePaths;
                    s2item.mask_area=mask_area;
                    s2item.label1 = label1;
                    s2item.sim1 = sim1;
                    q2dpu_s2.push(std::move(s2item));
                }
            }
            }
            ++count; if (count%cfg.json_flush_interval==0) flush();
        } else {
            if (semZ.try_acquire()){ flush(); break; }
        }
    }
}

// ============================================================================
// Thread 3 – DPU inference: run model and extract L2-normalised features
// ============================================================================

/**
 * @brief Pops preprocessed images, runs DPU inference, extracts features,
 *        L2-normalises them, and forwards the result to `queue3`.
 *
 * Signals `semZ` after the queue has been drained and `semAC` (from Thread 2)
 * has been raised.
 */
void dpuTask(SafeQueue<Image>& queue,  Semaphore& semAC,
             SafeQueue<Image>& queue3, Semaphore& semZ,
             std::unique_ptr<vart::RunnerExt>& runner,
             int& run_batch) {

    auto input_tensor_buffers  = runner->get_inputs();
    auto output_tensor_buffers = runner->get_outputs();
    CHECK_EQ(input_tensor_buffers.size(),  1u) << "expected single input tensor";
    CHECK_EQ(output_tensor_buffers.size(), 1u) << "expected single output tensor";

    auto output_tensor = output_tensor_buffers[0]->get_tensor();
    auto output_scale  = get_output_scale(output_tensor);

    // Log DPU output tensor shape for verification.
    const auto out_shape = output_tensor->get_shape();
    std::cout << "DPU output tensor shape: [";
    for (size_t i = 0; i < out_shape.size(); ++i)
        std::cout << out_shape[i] << (i + 1 < out_shape.size() ? ", " : "");
    std::cout << "]" << std::endl;
    // Clean, stable DPU loop
    {
        Image  preprocessed;
        int    count   = 0;
        double accumMs = 0.0;
        while (true) {
            if (queue.try_pop(preprocessed)) {
                ScopedTimer _t("DPU_Inference", "DPU", accumMs);
                for (int b = 0; b < run_batch; ++b) {
                    uint64_t data_in = 0u; size_t size_in = 0u;
                    std::tie(data_in, size_in) = input_tensor_buffers[0]->data(std::vector<int>{b,0,0,0});
                    assignImageDpu(preprocessed.batchImages[b], reinterpret_cast<void*>(data_in));
                }
                for (auto& inp : input_tensor_buffers) {
                    auto* t = inp->get_tensor();
                    auto sz = t->get_data_size() / std::max<int>(1, t->get_shape()[0]);
                    inp->sync_for_write(0, sz);
                }
                auto v = runner->execute_async(input_tensor_buffers, output_tensor_buffers);
                auto status = runner->wait(static_cast<int>(v.first), -1);
                CHECK_EQ(status, 0) << "DPU execution failed";
                for (auto& outp : output_tensor_buffers) {
                    auto* t = outp->get_tensor();
                    auto sz = t->get_data_size() / std::max<int>(1, t->get_shape()[0]);
                    outp->sync_for_read(0, sz);
                }
                auto feat = convert_fixpoint_to_float(output_tensor_buffers[0], output_scale, 0);
                l2_normalize(feat);
                preprocessed.features = std::move(feat);
                queue3.push(std::move(preprocessed));
                ++count;
            } else {
                if (semAC.try_acquire()) {
                    if (count > 0)
                        std::cout << "[Perf] DPU_Inference Avg: " << std::fixed << std::setprecision(2)
                                  << (accumMs / count) << " ms/img" << std::endl;
                    semZ.notify();
                    break;
                }
            }
        }
    }
    return;
}

#if 0
    DpuJob  job;
    int    count   = 0;
    double accumMs = 0.0;

    while (true) {
        // Prioritize Stage-2 jobs
        if (qfadpu_s2.try_pop(job)) {
            ScopedTimer _t("DPU_Inference(Stage2)", "DPU", accumMs);
            if (runner2) {
                // Build int8 from cropped_bgr
                cv::Mat s2 = preprocess_image(job.cropped_bgr, cv::Size(224,224));
                cv::Mat s2_int8 = setImageRGB_Stage2(s2, 8.f);
                uint64_t data_in = 0u; size_t size_in = 0u;
                std::tie(data_in, size_in) = input_tensor_buffers2[0]->data(std::vector<int>{0,0,0,0});
                assignImageDpu(s2_int8, reinterpret_cast<void*>(data_in));
                for (auto& inp : input_tensor_buffers2) {
                    Image  preprocessed;
                    auto sz = t->get_data_size() / std::max<int>(1, t->get_shape()[0]);
                    inp->sync_for_write(0, sz);
                }
                auto v = runner2->execute_async(input_tensor_buffers2, output_tensor_buffers2);
                        if (queue.try_pop(preprocessed)) {
                            ScopedTimer _t("DPU_Inference", "DPU", accumMs);
                            // Copy image data into the DPU input buffer.
                            for (int b = 0; b < run_batch; ++b) {
                                uint64_t data_in = 0u; size_t size_in = 0u;
                                std::tie(data_in, size_in) = input_tensor_buffers[0]->data(std::vector<int>{b,0,0,0});
                                assignImageDpu(preprocessed.batchImages[b], reinterpret_cast<void*>(data_in));
                            }
                            for (auto& inp : input_tensor_buffers) {
                                auto* t = inp->get_tensor();
                                auto sz = t->get_data_size() / std::max<int>(1, t->get_shape()[0]);
                                inp->sync_for_write(0, sz);
                            }
                            auto v = runner->execute_async(input_tensor_buffers, output_tensor_buffers);
                            auto status = runner->wait(static_cast<int>(v.first), -1);
                            CHECK_EQ(status, 0) << "DPU execution failed";
                            for (auto& outp : output_tensor_buffers) {
                                auto* t = outp->get_tensor();
                                auto sz = t->get_data_size() / std::max<int>(1, t->get_shape()[0]);
                                outp->sync_for_read(0, sz);
                            }
                            auto feat = convert_fixpoint_to_float(output_tensor_buffers[0], output_scale, 0);
                            l2_normalize(feat);
                            preprocessed.features = std::move(feat);
                            queue3.push(std::move(preprocessed));
                            ++count;

/**
 * @brief Pops L2-normalised feature vectors from `queue`, performs a FAISS
 *        k=1 nearest-neighbour search, and appends the result to the global
 *        `inferred_classes` / `ground_truth_classes` vectors.
                                semZ.notify();
 * METRIC_INNER_PRODUCT the inner product of two L2-normalised vectors equals
 * the cosine similarity, so no additional conversion is needed.
 */
#endif
void faissTask(SafeQueue<Image>& queue, Semaphore& semZ,
               const FaissDatabase& fdb1,
               const FaissDatabase& fdb2,
               std::unique_ptr<vart::RunnerExt>& runner2,
               const PipelineConfig& cfg,
               const std::map<std::string, std::pair<double,double>>& biomass_params) {

    constexpr int k = 1;          // k-nearest-neighbours
    float         distances[k];
    faiss::idx_t  indices[k];

    Image  item;
    int    count   = 0;
    double accumMs = 0.0;

    // JSON accumulators
    std::map<std::string, std::map<std::string, std::string>> areas_acc; // class -> {file: area}
    struct SpeciesEntry { std::string file; std::string predicted; bool has_bm=false; double bm=0.0; };
    std::map<std::string, std::map<std::string, std::vector<SpeciesEntry>>> species_acc;
    auto ensure_bucket = [&](const std::string& bucket){ if(!species_acc.count(bucket)) species_acc[bucket] = {}; };
    ensure_bucket("matched"); ensure_bucket("unmatched"); ensure_bucket("failed-cropping"); ensure_bucket("touches-border"); ensure_bucket("failed-segmentation"); ensure_bucket("unmatched-stage1");
    auto flush_jsons = [&](){
        // areas json
        try {
            fs::create_directories(fs::path(cfg.areas_json).parent_path());
            json_object* root = json_object_new_object();
            for (auto& [cls, mp] : areas_acc) {
                json_object* o = json_object_new_object();
                for (auto& [file, area] : mp) json_object_object_add(o, file.c_str(), json_object_new_string(area.c_str()));
                json_object_object_add(root, cls.c_str(), o);
            }
            std::ofstream ofs(cfg.areas_json); if (ofs) { ofs << json_object_to_json_string_ext(root, JSON_C_TO_STRING_PRETTY); ofs.close(); }
            json_object_put(root);
        } catch (...) {}
        // species json
        try {
            fs::create_directories(fs::path(cfg.species_json).parent_path());
            json_object* root = json_object_new_object();
            for (auto& bucket_pair : species_acc) {
                const std::string& bucket = bucket_pair.first; auto& clsmap = bucket_pair.second;
                json_object* b = json_object_new_object();
                for (auto& [cls, vec] : clsmap) {
                    json_object* arr = json_object_new_array();
                    for (auto& e : vec) {
                        json_object* ent = json_object_new_object();
                        json_object_object_add(ent, "file", json_object_new_string(e.file.c_str()));
                        if (!e.predicted.empty()) json_object_object_add(ent, "predicted", json_object_new_string(e.predicted.c_str())); else json_object_object_add(ent, "predicted", json_object_new_null());
                        if (e.has_bm) json_object_object_add(ent, "biomass", json_object_new_double(e.bm));
                        json_object_array_add(arr, ent);
                    }
                    json_object_object_add(b, cls.c_str(), arr);
                }
                json_object_object_add(root, bucket.c_str(), b);
            }
            std::ofstream ofs(cfg.species_json); if (ofs) { ofs << json_object_to_json_string_ext(root, JSON_C_TO_STRING_PRETTY); ofs.close(); }
            json_object_put(root);
        } catch (...) {}
    };

    int completed = 0; int flush_counter = 0; int total = 0; // total is approximate here; final flush at end anyway

    while (true) {
        if (queue.try_pop(item)) {
            {
                ScopedTimer _t("FAISS_Search(Stage1)", "FAISS", accumMs);
                // search() expects a row-major float matrix [nq × d].
                fdb1.index->search(1, item.features.data(), k, distances, indices);
            }

            const int         best_idx = static_cast<int>(indices[0]);
            const std::string label    = (best_idx >= 0 && best_idx < fdb1.ntotal)
                                             ? fdb1.labels[best_idx]
                                             : "unknown";

            // Decode numeric class id from the label prefix and compare as integers
            auto to_int = [](const std::string& s){ try { return std::stoi(s.substr(0, s.find('_'))); } catch(...) { return -1; } };
            int inferred_class_id = to_int(label);
            inferred_classes.push_back(inferred_class_id);
            ground_truth_classes.push_back(item.batchLabels[0]);
            // JSON bookkeeping
            std::string class_id, image_id; std::tie(class_id, image_id) = get_subtitles(item.batchImagePaths[0]);
            // areas_json always updated with mm² string using pixel_to_mm2 factor directly from YAML (already converted)
            {
                char buf[64]; std::snprintf(buf, sizeof(buf), "%.3f", item.mask_area * cfg.pixel_to_mm2);
                areas_acc[class_id][image_id] = std::string(buf);
            }
            // Stage-1 gating
            if (to_int(label) != item.batchLabels[0]) {
                species_acc["unmatched-stage1"][class_id].push_back({image_id, label, false, 0.0});
                ++completed; ++flush_counter;
            } else if (!item.seg_ok) {
                species_acc["failed-segmentation"][class_id].push_back({image_id, std::string(), false, 0.0});
                ++completed; ++flush_counter;
            } else if (!item.touch_ok) {
                species_acc["touches-border"][class_id].push_back({image_id, label, false, 0.0});
                ++completed; ++flush_counter;
            } else {
                // Stage-2 inline: DPU + FAISS-2
                if (!item.cropped_bgr.empty() && runner2) {
                    cv::Mat s2 = preprocess_image(item.cropped_bgr, cv::Size(224,224));
                    cv::Mat s2_int8 = setImageRGB_Stage2(s2, 8.f);
                    auto in_bufs  = runner2->get_inputs();
                    auto out_bufs = runner2->get_outputs();
                    uint64_t data_in = 0u; size_t size_in = 0u;
                    std::tie(data_in, size_in) = in_bufs[0]->data(std::vector<int>{0,0,0,0});
                    assignImageDpu(s2_int8, reinterpret_cast<void*>(data_in));
                    for (auto& inp : in_bufs) { auto* t=inp->get_tensor(); auto sz=t->get_data_size()/std::max<int>(1,t->get_shape()[0]); inp->sync_for_write(0, sz);} 
                    auto v = runner2->execute_async(in_bufs, out_bufs);
                    auto status = runner2->wait(static_cast<int>(v.first), -1);
                    CHECK_EQ(status, 0) << "DPU Stage-2 execution failed";
                    for (auto& outp : out_bufs) { auto* t=outp->get_tensor(); auto sz=t->get_data_size()/std::max<int>(1,t->get_shape()[0]); outp->sync_for_read(0, sz);} 
                    float out_scale2 = get_output_scale(out_bufs[0]->get_tensor());
                    auto feat2 = convert_fixpoint_to_float(out_bufs[0], out_scale2, 0);
                    l2_normalize(feat2);
                    float d2[k]; faiss::idx_t i2[k]; fdb2.index->search(1, feat2.data(), k, d2, i2);
                    std::string label2 = (i2[0] >= 0 && i2[0] < fdb2.ntotal) ? fdb2.labels[i2[0]] : std::string("unknown");
                    auto to_int2 = to_int;
                    if (to_int2(label2) == to_int(label)) {
                        double area_mm2 = item.mask_area * cfg.pixel_to_mm2;
                        double bm = 0.0; bool has_bm=false;
                        auto it = biomass_params.find(label2);
                        if (it != biomass_params.end()) { auto [a,b] = it->second; bm = a * std::pow(area_mm2, b); has_bm=true; }
                        species_acc["matched"][class_id].push_back({image_id, label2, has_bm, has_bm?bm:0.0});
                    } else {
                        species_acc["unmatched"][class_id].push_back({image_id, label2, false, 0.0});
                    }
                    ++completed; ++flush_counter;
                } else {
                    species_acc["failed-cropping"][class_id].push_back({image_id, std::string(), false, 0.0});
                    ++completed; ++flush_counter;
                }
            }

            if (cfg.json_flush_interval > 0 && flush_counter >= cfg.json_flush_interval) { flush_jsons(); flush_counter = 0; }
            ++count;
        } else {
            if (semZ.try_acquire()) {
                if (count > 0)
                    std::cout << "[Perf] FAISS_Search Avg: " << std::fixed << std::setprecision(2)
                              << (accumMs / count) << " ms/img" << std::endl;
                // Final flush
                flush_jsons();
                break;
            }
        }
    }
}

// ============================================================================
// Metric computation via embedded Python
// ============================================================================

/**
 * @brief Call compute_metrics.compute_metrics(gt, pred, num_classes) via the
 *        Python C API and print accuracy / precision / recall / F1.
 */
void compute_metrics(const std::vector<int>& gt,
                     const std::vector<int>& pred,
                     int                     num_classes) {
    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('.')");

    PyObject* pName   = PyUnicode_DecodeFSDefault("compute_metrics");
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule) {
        PyObject* pFunc = PyObject_GetAttrString(pModule, "compute_metrics");
        if (PyCallable_Check(pFunc)) {
            PyObject* pGT   = PyList_New(gt.size());
            PyObject* pPred = PyList_New(pred.size());
            for (size_t i = 0; i < gt.size(); ++i) {
                PyList_SetItem(pGT,   i, PyLong_FromLong(gt[i]));
                PyList_SetItem(pPred, i, PyLong_FromLong(pred[i]));
            }
            PyObject* pNC   = PyLong_FromLong(num_classes);
            PyObject* pArgs = PyTuple_Pack(3, pGT, pPred, pNC);
            PyObject* pVal  = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);

            if (pVal) {
                const double acc  = PyFloat_AsDouble(PyTuple_GetItem(pVal, 0));
                const double prec = PyFloat_AsDouble(PyTuple_GetItem(pVal, 1));
                const double rec  = PyFloat_AsDouble(PyTuple_GetItem(pVal, 2));
                const double f1   = PyFloat_AsDouble(PyTuple_GetItem(pVal, 3));
                std::cout << std::fixed << std::setprecision(2);
                std::cout << "Accuracy:  " << acc  * 100.0 << "%" << std::endl;
                std::cout << "Precision: " << prec * 100.0 << "%" << std::endl;
                std::cout << "Recall:    " << rec  * 100.0 << "%" << std::endl;
                std::cout << "F1 Score:  " << f1   * 100.0 << "%" << std::endl;
                Py_DECREF(pVal);
            } else {
                PyErr_Print();
            }
            Py_DECREF(pGT);
            Py_DECREF(pPred);
            Py_DECREF(pNC);
        } else {
            PyErr_Print();
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    } else {
        PyErr_Print();
    }
    Py_Finalize();
}

// ============================================================================
// Fixed-point → float conversion (DPU output)
// ============================================================================

/**
 * @brief Dequantise a DPU int8 output tensor slice to float.
 *
 * @param tensor_buffer  VART tensor buffer.
 * @param scale          Output scale = 2^(-fix_point).
 * @param batch_idx      Batch element index to extract.
 * @return Float vector of length `size` (= number of output neurons).
 */
static std::vector<float> convert_fixpoint_to_float(vart::TensorBuffer* tensor_buffer,
                                                     float scale, int batch_idx) {
    uint64_t data = 0u;
    size_t   size = 0u;
    std::tie(data, size) = tensor_buffer->data(std::vector<int>{batch_idx, 0});
    const signed char* data_c = reinterpret_cast<const signed char*>(data);
    std::vector<float> ret(size);
    std::transform(data_c, data_c + size, ret.begin(),
                   [scale](signed char v) { return static_cast<float>(v) * scale; });
    return ret;
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char* argv[]) {
    // Support new YAML-based CLI: --config <yaml> [limit] [nprobe]
    bool use_yaml = (argc >= 3 && std::string(argv[1]) == "--config");
    PipelineConfig cfg;
    std::string xmodel_file;
    std::string rootFolder;
    std::string faiss_path;
    std::string labels_db;
    int limit = 0;
    int nprobe = DEFAULT_NPROBE;

    if (use_yaml) {
        const std::string yaml_path = argv[2];
        cfg = load_config_from_yaml(yaml_path);
        // Reuse Stage-1 only pipeline below for now, but feed paths from YAML
        xmodel_file = cfg.ckpt_stage_1;
        rootFolder  = cfg.dataset_root_path;
        faiss_path  = cfg.faiss_index_stage_1;
        labels_db   = cfg.sqlite_db_stage_1;
        if (argc > 3) limit  = (std::isdigit(argv[3][0]) ? std::stoi(argv[3]) : 0);
        if (argc > 4) nprobe = (std::isdigit(argv[4][0]) ? std::stoi(argv[4]) : DEFAULT_NPROBE);
    } else {
        if (argc < 5) {
            std::cout
                << "Usage: " << argv[0]
                << " <backbone.xmodel> <folder_path>"
                   " <faiss_index.bin> <labels.sqlite> [limit] [nprobe]\n\n"
                << "  backbone.xmodel  : xmodel outputting 2048-dim feature vectors\n"
                << "  folder_path      : root of test images (class subdirs, e.g. 42_name/)\n"
                << "                     tip: copy to /dev/shm/ for max throughput\n"
                << "  faiss_index.bin  : native FAISS index (FlatIP → auto-converts to IVFPQ)\n"
                << "  labels.sqlite    : SQLite database with feature_mappings table\n"
                << "  limit  (optional): max images to process (0 = all)\n"
                << "  nprobe (optional): IVFPQ cells to probe (default " << DEFAULT_NPROBE << ")\n";
            return 0;
        }
        xmodel_file = std::string(argv[1]);
        rootFolder  = std::string(argv[2]);
        faiss_path  = std::string(argv[3]);
        labels_db   = std::string(argv[4]);
        limit       = (argc > 5 && std::isdigit(argv[5][0])) ? std::stoi(argv[5]) : 0;
        if (argc > 6 && std::isdigit(argv[6][0])) nprobe = std::stoi(argv[6]);
    }

    // Set the global epoch for relative tracing timestamps.
    g_epoch = std::chrono::high_resolution_clock::now();

    // ── 1. Load FAISS indexes (Stage-1 mandatory, Stage-2 optional) ───────
    FaissDatabase fdb1; fdb1.nprobe = nprobe;
    if (!loadFaissDatabase(faiss_path, labels_db, fdb1, cfg.faiss_ivfpq)) {
        std::cerr << "Failed to load FAISS Stage-1 database." << std::endl;
        return 1;
    }
    FaissDatabase fdb2; fdb2.nprobe = nprobe;
    bool have_stage2 = false;
    if (!cfg.faiss_index_stage_2.empty() && !cfg.sqlite_db_stage_2.empty()) {
        if (loadFaissDatabase(cfg.faiss_index_stage_2, cfg.sqlite_db_stage_2, fdb2, cfg.faiss_ivfpq)) have_stage2 = true;
        else std::cerr << "Warning: couldn't load Stage-2 FAISS/SQLite. Stage-2 disabled." << std::endl;
    }

    // ── 2. Create DPU runner ───────────────────────────────────────────────
    auto            graph    = xir::Graph::deserialize(xmodel_file);
    auto            root     = graph->get_root_subgraph();
    xir::Subgraph*  subgraph = nullptr;
    for (auto c : root->children_topological_sort()) {
        CHECK(c->has_attr("device"));
        if (c->get_attr<std::string>("device") == "DPU") { subgraph = c; break; }
    }
    auto attrs = xir::Attrs::create();
    std::unique_ptr<vart::RunnerExt> runner =
        vart::RunnerExt::create_runner(subgraph, attrs.get());
    // Stage-2 runner (optional)
    std::unique_ptr<xir::Graph> graph2;
    std::unique_ptr<vart::RunnerExt> runner2;
    if (have_stage2) {
        graph2 = xir::Graph::deserialize(cfg.ckpt_stage_2);
        auto            root2     = graph2->get_root_subgraph();
        xir::Subgraph*  subgraph2 = nullptr;
        for (auto c : root2->children_topological_sort()) {
            CHECK(c->has_attr("device"));
            if (c->get_attr<std::string>("device") == "DPU") { subgraph2 = c; break; }
        }
        auto attrs2 = xir::Attrs::create();
        runner2 = vart::RunnerExt::create_runner(subgraph2, attrs2.get());
    }

    int run_batch = 1;

    // ── 3. Allocate pipeline queues & semaphores (alternating) ────────────
    SafeQueue<Image> queue1(cfg.q12_max);  // Producer  → Consumer (bounded)
    SafeQueue<Image> q2dpu_s1(1);          // Consumer  → DPU (Stage-1)
    SafeQueue<Image> q2dpu_s2(1);          // FAISS     → DPU (Stage-2)
    SafeQueue<Image> qdpu_s1(1);           // DPU       → FAISS (Stage-1)
    SafeQueue<Image> qdpu_s2(1);           // DPU       → FAISS (Stage-2)
    Semaphore semB(0);        // T1 done signal  (→ T2)
    Semaphore semAC(0);       // T2 done signal  (→ T3)
    Semaphore semZ(0);        // T3 done signal  (→ T4)
    Semaphore next_image_sem(1); // controls per-image alternation

    // ── 4. Start all four pipeline threads ────────────────────────────────
    const auto start = std::chrono::high_resolution_clock::now();

    std::thread t1(imageProducer,
                   std::ref(queue1), std::ref(semB),
                   std::ref(limit),  std::ref(rootFolder),
                   cfg.show_progress_bar);

    std::thread t2(imageConsumer,
                   std::ref(queue1), std::ref(q2dpu_s1),
                   std::ref(semB),   std::ref(semAC), std::ref(next_image_sem),
                   std::ref(run_batch));

    std::thread t3(dpuTaskMux,
                   std::ref(q2dpu_s1), std::ref(q2dpu_s2),
                   std::ref(semAC), std::ref(semZ),
                   std::ref(qdpu_s1), std::ref(qdpu_s2),
                   std::ref(runner), std::ref(runner2), std::ref(run_batch),
                   std::ref(next_image_sem));

    // Load biomass parameters
    auto biomass_params = load_biomass_params(cfg.lehette_params);
    std::thread t4(faissTaskMux,
                   std::ref(qdpu_s1), std::ref(qdpu_s2),
                   std::ref(q2dpu_s2), std::ref(semZ),
                   std::cref(fdb1), std::cref(fdb2),
                   std::cref(cfg), std::cref(biomass_params), std::ref(next_image_sem));

    t1.join(); t2.join(); t3.join(); t4.join();

    // ── 5. Timing summary ─────────────────────────────────────────────────
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> diff = end - start;
    std::cout << "\nTotal images processed: " << limit    << std::endl;
    std::cout << "Average FPS: "
              << std::fixed << std::setprecision(1)
              << (limit / diff.count())                   << std::endl;

    // ── 6. Remap class ids to consecutive 0-based integers ────────────────
    std::set<int> unique_set(ground_truth_classes.begin(), ground_truth_classes.end());
    unique_set.insert(inferred_classes.begin(), inferred_classes.end());

    std::map<int, int> class_mapping;
    int new_label = 0;
    for (int old_label : unique_set) class_mapping[old_label] = ++new_label;

    for (int& l : ground_truth_classes) l = class_mapping[l] - 1;
    for (int& l : inferred_classes)     l = class_mapping[l] - 1;

    const int num_classes = static_cast<int>(class_mapping.size());
    if (cfg.compute_metrics)
        compute_metrics(ground_truth_classes, inferred_classes, num_classes);

    // ── 7. Dump Chrome Tracing JSON ────────────────────────────────────────
    Profiler::instance().dump("trace.json");

    std::cout << "Done." << std::endl;
    // Use _Exit to avoid potential hangs from Python/DPU destructors.
    std::_Exit(0);
}
