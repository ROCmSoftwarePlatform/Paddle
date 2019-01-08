/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/gpu_info.h"

#include <algorithm>

#include "gflags/gflags.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/split.h"

DEFINE_double(fraction_of_gpu_memory_to_use, 0.92,
              "Default use 92% of GPU memory for PaddlePaddle,"
              "reserve the rest for page tables, etc");

DEFINE_bool(enable_cublas_tensor_op_math, false,
            "The enable_cublas_tensor_op_math indicate whether to use Tensor Core, "
            "but it may loss precision. Currently, There are two CUDA libraries that"
            " use Tensor Cores, cuBLAS and cuDNN. cuBLAS uses Tensor Cores to speed up"
            " GEMM computations(the matrices must be either half precision or single "
            "precision); cuDNN uses Tensor Cores to speed up both convolutions(the "
            "input and output must be half precision) and recurrent neural networks "
            "(RNNs).");

DEFINE_string(selected_gpus, "",
	      "A list of device ids separated by comma, like: 0,1,2,3. "
	      "This option is useful when doing multi process training and "
	      "each process have only one device (GPU). If you want to use "
	      "all visible devices, set this to empty string. NOTE: the "
	      "reason of doing this is that we want to use P2P communication"
	      "between GPU devices, use CUDA_VISIBLE_DEVICES can only use"
	      "share-memory only.");

namespace paddle {
namespace platform {

int GetCUDADeviceCount() {
  int count;
  PADDLE_ENFORCE(
      hipGetDeviceCount(&count),
      "hipGetDeviceCount failed in paddle::platform::GetCUDADeviceCount");
  return count;
}

int GetCUDAComputeCapability(int id) {
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(), "id must less than GPU count");
  hipDeviceProp_t device_prop;
  PADDLE_ENFORCE(hipGetDeviceProperties(&device_prop, id),
                 "hipGetDeviceProperties failed in "
                 "paddle::platform::GetCUDAComputeCapability");
  return device_prop.major * 10 + device_prop.minor;
}

int GetCUDAMultiProcessors(int id) {
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(), "id must less than GPU count");
  int count;
  PADDLE_ENFORCE(
      hipDeviceGetAttribute(&count, hipDeviceAttributeMultiprocessorCount, id),
      "hipDeviceGetAttribute failed in "
      "paddle::platform::GetCUDAMultiProcessors");
  return count;
}

int GetCUDAMaxThreadsPerMultiProcessor(int id) {
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(), "id must less than GPU count");
  int count;
  PADDLE_ENFORCE(hipDeviceGetAttribute(
                     &count, hipDeviceAttributeMaxThreadsPerMultiProcessor, id),
                 "hipDeviceGetAttribute failed in "
                 "paddle::platform::GetCUDAMaxThreadsPerMultiProcessor");
  return count;
}

int GetCurrentDeviceId() {
  int device_id;
  PADDLE_ENFORCE(
      hipGetDevice(&device_id),
      "hipGetDevice failed in paddle::platform::GetCurrentDeviceId");
  return device_id;
}

//! Get a list of device ids from environment variable or use all.
std::vector<int> GetSelectedDevices() {
  // use user specified GPUs in single-node multi-process mode.
  std::vector<int> devices;
  if (!FLAGS_selected_gpus.empty()) {
    auto devices_str = paddle::string::Split(FLAGS_selected_gpus, ',');
    for (auto id : devices_str) {
      devices.push_back(atoi(id.c_str()));
    }
  } else {
    int count = GetCUDADeviceCount();
    for (int i = 0; i < count; ++i) {
      devices.push_back(i);
    }
  }
  return devices;
}



void SetDeviceId(int id) {
  // TODO(qijun): find a better way to cache the cuda device count
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(), "id must less than GPU count");
  PADDLE_ENFORCE(hipSetDevice(id),
                 "hipSetDevice failed in paddle::platform::SetDeviceId");
}

void GpuMemoryUsage(size_t* available, size_t* total) {
  PADDLE_ENFORCE(hipMemGetInfo(available, total),
                 "hipMemGetInfo failed in paddle::platform::GetMemoryUsage");
}

size_t GpuMaxAllocSize() {
  size_t total = 0;
  size_t available = 0;

  GpuMemoryUsage(&available, &total);

  // Reserve the rest for page tables, etc.
  return static_cast<size_t>(total * FLAGS_fraction_of_gpu_memory_to_use);
}

size_t GpuMinChunkSize() {
  // Allow to allocate the minimum chunk size is 256 bytes.
  return 1 << 8;
}

size_t GpuMaxChunkSize() {
  size_t total = 0;
  size_t available = 0;

  GpuMemoryUsage(&available, &total);
  VLOG(10) << "GPU Usage " << available / 1024 / 1024 << "M/"
           << total / 1024 / 1024 << "M";
  size_t reserving = static_cast<size_t>(0.05 * total);
  // If available less than minimum chunk size, no usable memory exists.
  available =
      std::min(std::max(available, GpuMinChunkSize()) - GpuMinChunkSize(),
               total - reserving);

  // Reserving the rest memory for page tables, etc.

  size_t allocating = static_cast<size_t>(FLAGS_fraction_of_gpu_memory_to_use *
                                          (total - reserving));

  PADDLE_ENFORCE_LE(allocating, available);

  return allocating;
}

void GpuMemcpyAsync(void *dst, const void *src, size_t count,
                    enum hipMemcpyKind kind, hipStream_t stream) {
  PADDLE_ENFORCE(hipMemcpyAsync(dst, src, count, kind, stream),
                 "hipMemcpyAsync failed in paddle::platform::GpuMemcpyAsync");
}

void GpuMemcpySync(void *dst, const void *src, size_t count,
                   enum hipMemcpyKind kind) {
  PADDLE_ENFORCE(hipMemcpy(dst, src, count, kind),
                 "hipMemcpy failed in paddle::platform::GpuMemcpySync");
}

void GpuMemcpyPeerAsync(void *dst, int dst_device, const void *src,
                        int src_device, size_t count, hipStream_t stream) {
  PADDLE_ENFORCE(
      hipMemcpyPeerAsync(dst, dst_device, src, src_device, count, stream),
      "hipMemcpyPeerAsync failed in paddle::platform::GpuMemcpyPeerAsync");
}

void GpuMemcpyPeerSync(void *dst, int dst_device, const void *src,
                       int src_device, size_t count) {
  PADDLE_ENFORCE(
      hipMemcpyPeer(dst, dst_device, src, src_device, count),
      "cudaMemcpyPeer failed in paddle::platform::GpuMemcpyPeerSync");
}

void GpuMemsetAsync(void *dst, int value, size_t count, hipStream_t stream) {
  PADDLE_ENFORCE(hipMemsetAsync(dst, value, count, stream),
                 "hipMemsetAsync failed in paddle::platform::GpuMemsetAsync");
}
}  // namespace platform
}  // namespace paddle
