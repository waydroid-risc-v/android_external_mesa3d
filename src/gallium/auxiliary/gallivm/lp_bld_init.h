/**************************************************************************
 *
 * Copyright 2009 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/


#ifndef LP_BLD_INIT_H
#define LP_BLD_INIT_H

#define GALLIVM_USE_ORCJIT 1

#include "util/compiler.h"
#include "util/u_pointer.h" // for func_pointer
#include "lp_bld.h"
#if GALLIVM_USE_ORCJIT == 1
#include <llvm-c/Orc.h>
#else
#include <llvm-c/ExecutionEngine.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct lp_cached_code;
struct gallivm_state
{
   char *module_name;
   LLVMModuleRef module;
   LLVMTargetDataRef target;
#if GALLIVM_USE_ORCJIT == 1
   /* own this->module */
   LLVMOrcThreadSafeContextRef _ts_context;
   /* each module is in its own jitdylib */
   LLVMOrcJITDylibRef _per_module_jd;
#else
   LLVMExecutionEngineRef engine;
#if GALLIVM_USE_NEW_PASS == 0
   LLVMPassManagerRef passmgr;
#if GALLIVM_HAVE_CORO == 1
   LLVMPassManagerRef cgpassmgr;
#endif
#endif
   LLVMMCJITMemoryManagerRef memorymgr;
   struct lp_generated_code *code;
#endif

   LLVMContextRef context;
   LLVMBuilderRef builder;

   struct lp_cached_code *cache;
   unsigned compiled;
   LLVMValueRef coro_malloc_hook;
   LLVMValueRef coro_free_hook;
   LLVMValueRef debug_printf_hook;

   LLVMTypeRef coro_malloc_hook_type;
   LLVMTypeRef coro_free_hook_type;

   LLVMValueRef get_time_hook;

   LLVMValueRef texture_descriptor;
   LLVMValueRef sampler_descriptor;
};

unsigned
lp_build_init_native_width(void);

bool
lp_build_init(void);

#if GALLIVM_USE_ORCJIT == 1
struct gallivm_state *
gallivm_create(const char *name, LLVMOrcThreadSafeContextRef context,
               struct lp_cached_code *cache);
#else
struct gallivm_state *
gallivm_create(const char *name, LLVMContextRef context,
               struct lp_cached_code *cache);
#endif

void
gallivm_destroy(struct gallivm_state *gallivm);

void
gallivm_free_ir(struct gallivm_state *gallivm);

void
gallivm_verify_function(struct gallivm_state *gallivm,
                        LLVMValueRef func);

void
gallivm_add_global_mapping(struct gallivm_state *gallivm, LLVMValueRef sym, void* addr);

/**
 * for ORCJIT, after this function gets called, all access and modification to
 * module and any structure associated to it should be avoided,
 * as module has been moved into ORCJIT and may be recycled
 */
void
gallivm_compile_module(struct gallivm_state *gallivm);

#if GALLIVM_USE_ORCJIT == 1
func_pointer
gallivm_jit_function(struct gallivm_state *gallivm,
                     const char *func_name);
#else
func_pointer
gallivm_jit_function(struct gallivm_state *gallivm,
                     LLVMValueRef func);
#endif

unsigned gallivm_get_perf_flags(void);

void lp_init_clock_hook(struct gallivm_state *gallivm);
#ifdef __cplusplus
}
#endif

#endif /* !LP_BLD_INIT_H */
