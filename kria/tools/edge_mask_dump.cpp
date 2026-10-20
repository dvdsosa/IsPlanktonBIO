// Deterministic masks of the 120 manually outlined images, computed by the deployed KV260 code
// (data/manual_reference, paper Section 4.3 and Table 2). The TurboJPEG BGR decoder, the contrast adjustment and
// segment_and_crop() are copied verbatim from ../resnet50_cosine_faiss_turbojpeg_IVFPQ.cpp.
//
// Build and run on the KV260:
//   g++ -O2 -std=c++17 edge_mask_dump.cpp -o edge_mask_dump $(pkg-config --cflags --libs opencv4) -lturbojpeg
//   ./edge_mask_dump list.txt out     # list.txt: one image path per line, <DYB-PlanktonNet>/<class folder>/<file>.jpg
// Writes out/<class folder>__<stem>.png (filled mask) and out/areas.csv (reported contour-polygon area, mask pixels).
// data/manual_reference/scripts/edge_masks.py --kria-masks out checks that these masks match its own port.
#include <opencv2/opencv.hpp>
#include <turbojpeg.h>
#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>
#include <string>
#include <filesystem>
namespace fs = std::filesystem;

static cv::Mat decodeTurboJPEG(tjhandle tjHandle_, const std::string& path) {
    std::vector<unsigned char> fileBuf_;
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
            std::cerr << "TurboJPEG header error: " << tjGetErrorStr() << std::endl;
            return {};
        }

        // 3. Decompress directly into pre-allocated cv::Mat (BGR output).
        cv::Mat img(height, width, CV_8UC3);
        if (tjDecompress2(tjHandle_, fileBuf_.data(),
                          static_cast<unsigned long>(fileSize),
                          img.data, width, /*pitch=*/0, height,
                          TJPF_BGR,
                          TJFLAG_ACCURATEDCT | TJFLAG_NOREALLOC) != 0) {
            std::cerr << "TurboJPEG decompress error: " << tjGetErrorStr() << std::endl;
            return {};
        }
        return img;
}
static cv::Mat adjust_brightness_contrast_cv(const cv::Mat& gray, int brightness, int contrast) {
    // alpha = contrast/100, beta = (brightness-100)*2.55
    double alpha = static_cast<double>(contrast) / 100.0;
    double beta  = static_cast<double>(brightness - 100) * 2.55;
    cv::Mat out; cv::convertScaleAbs(gray, out, alpha, beta);
    return out;
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

int main(int argc, char* argv[]) {
    if (argc < 3) { std::cerr << "usage: edge_mask_dump list.txt out_dir\n"; return 1; }
    std::ifstream lst(argv[1]); std::string out = argv[2]; fs::create_directories(out);
    std::ofstream csv(out + "/areas.csv"); csv << "file,ok,mask_area_poly,mask_px\n";
    tjhandle h = tjInitDecompress();
    std::string p; int n = 0;
    while (std::getline(lst, p)) {
        if (p.empty()) continue;
        cv::Mat bgr = decodeTurboJPEG(h, p);
        auto [ok, area, mask, cropped] = segment_and_crop(bgr);
        std::string base = fs::path(p).parent_path().filename().string() + "__" + fs::path(p).stem().string();
        if (ok) cv::imwrite(out + "/" + base + ".png", mask);
        csv << base << "," << ok << "," << area << "," << (ok ? cv::countNonZero(mask) : 0) << "\n";
        ++n;
    }
    tjDestroy(h);
    std::cout << "processed " << n << " images\n";
    return 0;
}
