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
#include <rccl.h>
#include <mutex>
#include "paddle/fluid/platform/call_once.h"
#include "paddle/fluid/platform/dynload/dynamic_loader.h"

namespace paddle {
namespace platform {
namespace dynload {

extern std::once_flag rccl_dso_flag;
extern void* rccl_dso_handle;

#ifdef PADDLE_USE_DSO
extern void LoadRCCLDSO();

#define DECLARE_DYNAMIC_LOAD_RCCL_WRAP(__name)                   \
  struct DynLoad__##__name {                                     \
    template <typename... Args>                                  \
    auto operator()(Args... args) -> decltype(__name(args...)) { \
      using rccl_func = decltype(__name(args...)) (*)(Args...);  \
      paddle::platform::dynload::LoadRCCLDSO();                  \
      void* p_##__name = dlsym(rccl_dso_handle, #__name);        \
      return reinterpret_cast<rccl_func>(p_##__name)(args...);   \
    }                                                            \
  };                                                             \
  extern DynLoad__##__name __name
#else
#define DECLARE_DYNAMIC_LOAD_RCCL_WRAP(__name) \
  struct DynLoad__##__name {                   \
    template <typename... Args>                \
    rcclResult_t operator()(Args... args) {    \
      return __name(args...);                  \
    }                                          \
  };                                           \
  extern DynLoad__##__name __name
#endif

#define RCCL_RAND_ROUTINE_EACH(__macro) \
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
  __macro(rcclReduce);                  \
  __macro(rcclGetErrorString);

RCCL_RAND_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_RCCL_WRAP)

}  // namespace dynload
}  // namespace platform
}  // namespace paddle