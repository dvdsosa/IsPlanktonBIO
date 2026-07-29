# Building FAISS on the Kria KV260

`resnet50_cosine_faiss_turbojpeg_IVFPQ.cpp` links against FAISS (`-lfaiss`) and
OpenBLAS (`-lopenblas`). Neither ships as a prebuilt package for this
PetaLinux/aarch64 image, so both must be built from source on the board before
the [build task](.vscode/tasks.json) will succeed. Adapted from the board's own
`~/vai_runtime/docs/compilacion-faiss.md`.

1. Build OpenBLAS (no RPM package available for this target):

   ```bash
   cd /tmp
   git clone --depth 1 https://github.com/OpenMathLib/OpenBLAS.git
   cd OpenBLAS
   make TARGET=CORTEXA53 USE_THREAD=1 USE_OPENMP=0 NO_LAPACK=0 -j4
   sudo make TARGET=CORTEXA53 USE_THREAD=1 USE_OPENMP=0 NO_LAPACK=0 PREFIX=/usr/local install

   echo "/usr/local/lib" | sudo tee -a /etc/ld.so.conf
   sudo ldconfig
   ldconfig -p | grep openblas
   ```

2. Fetch FAISS (this project was built against v1.9.0):

   ```bash
   cd /tmp
   git clone --depth 1 --branch v1.9.0 https://github.com/facebookresearch/faiss.git
   cd faiss
   mkdir -p build && cd build
   ```

3. Edit `faiss/faiss/CMakeLists.txt` and comment out the following lines
   (OpenMP is unavailable/unneeded on this target):

   ```cpp
   # find_package(OpenMP REQUIRED)
   # target_link_libraries(faiss PRIVATE OpenMP::OpenMP_CXX)
   # target_link_libraries(faiss_avx2 PRIVATE OpenMP::OpenMP_CXX)
   # target_link_libraries(faiss_avx512 PRIVATE OpenMP::OpenMP_CXX)
   # target_link_libraries(faiss_sve PRIVATE OpenMP::OpenMP_CXX)
   ```

4. Configure and build FAISS:

   ```bash
   cmake .. \
       -DFAISS_ENABLE_OPENMP=OFF \
       -DFAISS_ENABLE_GPU=OFF \
       -DFAISS_ENABLE_PYTHON=OFF \
       -DFAISS_ENABLE_C_API=OFF \
       -DBUILD_TESTING=OFF \
       -DBUILD_SHARED_LIBS=ON \
       -DCMAKE_BUILD_TYPE=Release \
       -DBLA_VENDOR=OpenBLAS \
       -DFAISS_OPT_LEVEL=generic \
       -DCMAKE_INSTALL_PREFIX=/usr/local

   make -j2
   ```

Once both are installed under `/usr/local`, the build task in
[`.vscode/tasks.json`](.vscode/tasks.json) (which already points `-I`/`-L` at
`/usr/local/include` and `/usr/local/lib64`/`/usr/local/lib`) will find them.

---

[⬅ Back to main README](../README.md)
