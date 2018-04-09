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
#pragma once

#include <dlfcn.h>
#ifdef PADDLE_WITH_HIP
#include <rccl.h>
#else
#include <nccl.h>
#endif

#include <mutex>  // NOLINT

#include "paddle/fluid/platform/dynload/dynamic_loader.h"

namespace paddle {
namespace platform {
namespace dynload {

extern std::once_flag nccl_dso_flag;
extern void* nccl_dso_handle;

#ifdef PADDLE_USE_DSO

#define DECLARE_DYNAMIC_LOAD_NCCL_WRAP(__name)                           \
  struct DynLoad__##__name {                                             \
    template <typename... Args>                                          \
    auto operator()(Args... args) -> decltype(__name(args...)) {         \
      using nccl_func = decltype(&::__name);                             \
      std::call_once(nccl_dso_flag, []() {                               \
        nccl_dso_handle = paddle::platform::dynload::GetNCCLDsoHandle(); \
      });                                                                \
      static void* p_##__name = dlsym(nccl_dso_handle, #__name);         \
      return reinterpret_cast<nccl_func>(p_##__name)(args...);           \
    }                                                                    \
  };                                                                     \
  extern DynLoad__##__name __name
#else
#define DECLARE_DYNAMIC_LOAD_NCCL_WRAP(__name) \
  struct DynLoad__##__name {                   \
    template <typename... Args>                \
    rcclResult_t operator()(Args... args) {    \
      return __name(args...);                  \
    }                                          \
  };                                           \
  extern DynLoad__##__name __name
#endif

#define NCCL_RAND_ROUTINE_EACH(__macro) \
  __macro(rcclCommInitAll);             \
  __macro(rcclGetUniqueId);             \
  __macro(rcclCommInitRank);            \
  __macro(rcclCommDestroy);             \
  __macro(rcclCommCount);               \
  __macro(rcclCommCuDevice);            \
  __macro(rcclCommUserRank);            \
  __macro(rcclAllReduce);               \
  __macro(rcclBcast);                   \
  __macro(rcclAllGather);               \
  __macro(rcclGroupStart);              \
  __macro(rcclGroupEnd);                \
  __macro(rcclReduce);                  \
  __macro(rcclGetErrorString);

NCCL_RAND_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_NCCL_WRAP)

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
