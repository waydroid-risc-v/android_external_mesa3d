#include "util/detect.h"
#include "util/u_cpu_detect.h"
#include "util/u_debug.h"
#include "util/os_time.h"
#include "lp_bld.h"
#include "lp_bld_debug.h"
#include "lp_bld_init.h"

#include <llvm/Config/llvm-config.h>
#include <llvm-c/Core.h>
#include <llvm-c/Orc.h>
#include <llvm-c/LLJIT.h>
#include <llvm-c/TargetMachine.h>
#include <llvm-c/Support.h>

#include <llvm-c/Analysis.h>
#if LLVM_VERSION_MAJOR < 17
#include <llvm-c/Transforms/Scalar.h>
#if LLVM_VERSION_MAJOR >= 7
#include <llvm-c/Transforms/Utils.h>
#endif
#endif
#include <llvm-c/BitWriter.h>
#if GALLIVM_USE_NEW_PASS == 1
#include <llvm-c/Transforms/PassBuilder.h>
#elif GALLIVM_HAVE_CORO == 1
#if LLVM_VERSION_MAJOR <= 8 && (defined(PIPE_ARCH_AARCH64) || defined (PIPE_ARCH_ARM) || defined(PIPE_ARCH_S390) || defined(PIPE_ARCH_MIPS64))
#include <llvm-c/Transforms/IPO.h>
#endif
#include <llvm-c/Transforms/Coroutines.h>
#endif

#include <llvm/ADT/StringMap.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include <llvm/Target/TargetMachine.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/CBindingWrapping.h>
#if LLVM_USE_INTEL_JITEVENTS
#include <llvm/ExecutionEngine/JITEventListener.h>
#endif

/* conflict with ObjectLinkingLayer.h */
#include "util/u_memory.h"

#if defined(PIPE_ARCH_RISCV64) || defined(PIPE_ARCH_RISCV32) || (defined(_WIN32) && LLVM_VERSION_MAJOR >= 15)
/* use ObjectLinkingLayer (JITLINK backend) */
#define USE_JITLINK
#endif
/* else use old RTDyldObjectLinkingLayer (RuntimeDyld backend) */

unsigned lp_native_vector_width;

unsigned gallivm_perf = 0;

static const struct debug_named_value lp_bld_perf_flags[] = {
   { "brilinear", GALLIVM_PERF_BRILINEAR, "enable brilinear optimization" },
   { "rho_approx", GALLIVM_PERF_RHO_APPROX, "enable rho_approx optimization" },
   { "no_quad_lod", GALLIVM_PERF_NO_QUAD_LOD, "disable quad_lod optimization" },
   { "no_aos_sampling", GALLIVM_PERF_NO_AOS_SAMPLING, "disable aos sampling optimization" },
   { "nopt",   GALLIVM_PERF_NO_OPT, "disable optimization passes to speed up shader compilation" },
   DEBUG_NAMED_VALUE_END
};

unsigned gallivm_debug = 0;

static const struct debug_named_value lp_bld_debug_flags[] = {
   { "tgsi",   GALLIVM_DEBUG_TGSI, NULL },
   { "ir",     GALLIVM_DEBUG_IR, NULL },
   { "asm",    GALLIVM_DEBUG_ASM, NULL },
   { "perf",   GALLIVM_DEBUG_PERF, NULL },
   { "gc",     GALLIVM_DEBUG_GC, NULL },
   { "dumpbc", GALLIVM_DEBUG_DUMP_BC, NULL },
   DEBUG_NAMED_VALUE_END
};

DEBUG_GET_ONCE_FLAGS_OPTION(gallivm_debug, "GALLIVM_DEBUG", lp_bld_debug_flags, 0)

struct lp_cached_code {
   void *data;
   size_t data_size;
   bool dont_cache;
   void *jit_obj_cache;
};

namespace {

class LPJit;

class LLVMEnsureMultithreaded {
public:
   LLVMEnsureMultithreaded()
   {
      if (!LLVMIsMultithreaded()) {
         LLVMStartMultithreaded();
      }
   }
};

LLVMEnsureMultithreaded lLVMEnsureMultithreaded;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(llvm::orc::ThreadSafeContext,
                                   LLVMOrcThreadSafeContextRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(llvm::orc::IRTransformLayer,
                                   LLVMOrcIRTransformLayerRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(llvm::orc::JITDylib, LLVMOrcJITDylibRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(llvm::orc::JITTargetMachineBuilder,
                                   LLVMOrcJITTargetMachineBuilderRef)
LLVMTargetMachineRef wrap(const llvm::TargetMachine *P) {
   return reinterpret_cast<LLVMTargetMachineRef>(const_cast<llvm::TargetMachine*>(P));
}

llvm::ExitOnError ExitOnErr;

inline const char* get_module_name(LLVMModuleRef mod) {
   using llvm::Module;
   return llvm::unwrap(mod)->getModuleIdentifier().c_str();
}

once_flag init_lpjit_once_flag = ONCE_FLAG_INIT;

/* A JIT singleton built upon LLJIT */
class LPJit
{
public:
   static LPJit* get_instance() {
      call_once(&init_lpjit_once_flag, init_lpjit);
      return jit;
   }

   gallivm_state *find_gallivm_state(LLVMModuleRef mod) {
#if DEBUG
      using llvm::Module;
      auto I = gallivm_modules.find(llvm::unwrap(mod)->getModuleIdentifier());
      if (I == gallivm_modules.end()) {
         debug_printf("No gallivm state found for module: %s", get_module_name(mod));
         return NULL;
      }
      return I->second;
#endif
      return NULL;
   }

   static char *get_unique_name(const char* name) {
      LPJit* jit = get_instance();
      size_t size = name == NULL? 16: strlen(name) + 16;
      char *name_uniq = (char *)MALLOC(size);
      if (!name_uniq) {
         return NULL;
      }
      do {
         snprintf(name_uniq, size, "%s_%u", name, jit->jit_dylib_count++);
      } while(jit->lljit->getExecutionSession().getJITDylibByName(name_uniq));
      return name_uniq;
   }

   static LLVMOrcJITDylibRef create_jit_dylib(const char * name) {
      using llvm::orc::JITDylib;
      LPJit* jit = get_instance();
      JITDylib& tmp = ExitOnErr(jit->lljit->createJITDylib(name));
      return wrap(&tmp);
   }

   static void register_gallivm_state(gallivm_state *gallivm) {
#if DEBUG
      LPJit* jit = get_instance();
      jit->gallivm_modules[gallivm->module_name] = gallivm;
#endif
   }

   static void deregister_gallivm_state(gallivm_state *gallivm) {
#if DEBUG
      LPJit* jit = get_instance();
      (void)jit->gallivm_modules.erase(gallivm->module_name);
#endif
   }

   static void add_ir_module_to_jd(
         LLVMOrcThreadSafeContextRef ts_context,
         LLVMModuleRef mod,
         LLVMOrcJITDylibRef jd) {
      using llvm::Module;
      using llvm::orc::ThreadSafeModule;
      using llvm::orc::JITDylib;
      ThreadSafeModule tsm(
         std::unique_ptr<Module>(llvm::unwrap(mod)), *::unwrap(ts_context));
      ExitOnErr(get_instance()->lljit->addIRModule(
         *::unwrap(jd), std::move(tsm)
      ));
   }

   static void add_mapping_to_jd(
         LLVMValueRef sym,
         void *addr,
         LLVMOrcJITDylibRef jd) {
#if LLVM_VERSION_MAJOR >= 17
      using llvm::orc::ExecutorAddr;
      using llvm::orc::ExecutorSymbolDef;
      using llvm::JITSymbolFlags;
#else
      using llvm::JITEvaluatedSymbol;
#endif
      using llvm::orc::ExecutionSession;
      using llvm::orc::JITDylib;
      using llvm::orc::SymbolMap;
      JITDylib* JD = ::unwrap(jd);
      auto& es = LPJit::get_instance()->lljit->getExecutionSession();
      auto name = es.intern(llvm::unwrap(sym)->getName());
      SymbolMap map(1);
#if LLVM_VERSION_MAJOR >= 17
      map[name] = ExecutorSymbolDef(ExecutorAddr::fromPtr(addr), JITSymbolFlags::Exported);
#else
      map[name] = JITEvaluatedSymbol::fromPointer(addr);
#endif
      auto munit = llvm::orc::absoluteSymbols(map);
      llvm::cantFail(JD->define(std::move(munit)));
   }

   static void *lookup_in_jd(
         const char *func_name,
         LLVMOrcJITDylibRef jd) {
      using llvm::orc::JITDylib;
      using llvm::JITEvaluatedSymbol;
      using llvm::orc::ExecutorAddr;
      JITDylib* JD = ::unwrap(jd);
      auto func = ExitOnErr(LPJit::get_instance()->lljit->lookup(*JD, func_name));
#if LLVM_VERSION_MAJOR >= 16
      return func.toPtr<void *>();
#else
      return (void *)(func.getAddress());
#endif
   }

   static void remove_jd(LLVMOrcJITDylibRef jd) {
      using llvm::orc::ExecutionSession;
      using llvm::orc::JITDylib;
      auto& es = LPJit::get_instance()->lljit->getExecutionSession();
      ExitOnErr(es.removeJITDylib(* ::unwrap(jd)));
   }

   LLVMTargetMachineRef tm;

private:
   LPJit();
   ~LPJit() = default;
   LPJit(const LPJit&) = delete;
   LPJit& operator=(const LPJit&) = delete;

   static void init_native_targets();
   llvm::orc::JITTargetMachineBuilder create_jtdb();

   static void init_lpjit() {
      jit = new LPJit;
   }
   static LPJit* jit;

   std::unique_ptr<llvm::orc::LLJIT> lljit;
   /* avoid name conflict */
   unsigned jit_dylib_count;

#if DEBUG
   /* map from module name to gallivm_state */
   llvm::StringMap<gallivm_state *> gallivm_modules;
#endif
};

LPJit* LPJit::jit = NULL;

LLVMErrorRef module_transform(void *Ctx, LLVMModuleRef mod) {
   int64_t time_begin = 0;
   if (::gallivm_debug & GALLIVM_DEBUG_PERF)
      time_begin = os_time_get();
#if GALLIVM_USE_NEW_PASS == 1
   char passes[1024];
   passes[0] = 0;

   /*
    * there should be some way to combine these two pass runs but I'm not seeing it,
    * at the time of writing.
    */
   strcpy(passes, "default<O0>");

   LLVMPassBuilderOptionsRef opts = LLVMCreatePassBuilderOptions();
   LLVMRunPasses(mod, passes, LPJit::get_instance()->tm, opts);

   if (!(gallivm_perf & GALLIVM_PERF_NO_OPT))
      strcpy(passes, "sroa,early-cse,simplifycfg,reassociate,mem2reg,constprop,instcombine,");
   else
      strcpy(passes, "mem2reg");

   LLVMRunPasses(mod, passes, LPJit::get_instance()->tm, opts);
   LLVMDisposePassBuilderOptions(opts);

   return LLVMErrorSuccess;

#else /* GALLIVM_USE_NEW_PASS */
   LLVMPassManagerRef passmgr = LLVMCreateFunctionPassManagerForModule(mod);

#if GALLIVM_HAVE_CORO == 1
   LLVMPassManagerRef cgpassmgr = LLVMCreatePassManager();
#endif

#if GALLIVM_HAVE_CORO == 1
#if LLVM_VERSION_MAJOR <= 8 && (defined(PIPE_ARCH_AARCH64) || defined (PIPE_ARCH_ARM) || defined(PIPE_ARCH_S390) || defined(PIPE_ARCH_MIPS64))
   LLVMAddArgumentPromotionPass(cgpassmgr);
   LLVMAddFunctionAttrsPass(cgpassmgr);
#endif
   LLVMAddCoroEarlyPass(cgpassmgr);
   LLVMAddCoroSplitPass(cgpassmgr);
   LLVMAddCoroElidePass(cgpassmgr);
#endif

   if ((gallivm_perf & GALLIVM_PERF_NO_OPT) == 0) {
      /*
       * TODO: Evaluate passes some more - keeping in mind
       * both quality of generated code and compile times.
       */
      /*
       * NOTE: if you change this, don't forget to change the output
       * with GALLIVM_DEBUG_DUMP_BC in gallivm_compile_module.
       */
      LLVMAddScalarReplAggregatesPass(passmgr);
      LLVMAddEarlyCSEPass(passmgr);
      LLVMAddCFGSimplificationPass(passmgr);
      /*
       * FIXME: LICM is potentially quite useful. However, for some
       * rather crazy shaders the compile time can reach _hours_ per shader,
       * due to licm implying lcssa (since llvm 3.5), which can take forever.
       * Even for sane shaders, the cost of licm is rather high (and not just
       * due to lcssa, licm itself too), though mostly only in cases when it
       * can actually move things, so having to disable it is a pity.
       * LLVMAddLICMPass(passmgr);
       */
      LLVMAddReassociatePass(passmgr);
      LLVMAddPromoteMemoryToRegisterPass(passmgr);
#if LLVM_VERSION_MAJOR <= 11
      LLVMAddConstantPropagationPass(passmgr);
#else
      LLVMAddInstructionSimplifyPass(passmgr);
#endif
      LLVMAddInstructionCombiningPass(passmgr);
      LLVMAddGVNPass(passmgr);
   }
   else {
      /* We need at least this pass to prevent the backends to fail in
       * unexpected ways.
       */
      LLVMAddPromoteMemoryToRegisterPass(passmgr);
   }
#if GALLIVM_HAVE_CORO == 1
   LLVMAddCoroCleanupPass(passmgr);

   LLVMRunPassManager(cgpassmgr, mod);
#endif
   /* Run optimization passes */
   LLVMInitializeFunctionPassManager(passmgr);
   LLVMValueRef func;
   func = LLVMGetFirstFunction(mod);
   while (func) {
      if (0) {
         debug_printf("optimizing func %s...\n", LLVMGetValueName(func));
      }

   /* Disable frame pointer omission on debug/profile builds */
   /* XXX: And workaround http://llvm.org/PR21435 */
#if defined(DEBUG) || defined(PROFILE) || defined(PIPE_ARCH_X86) || defined(PIPE_ARCH_X86_64)
      LLVMAddTargetDependentFunctionAttr(func, "no-frame-pointer-elim", "true");
      LLVMAddTargetDependentFunctionAttr(func, "no-frame-pointer-elim-non-leaf", "true");
#endif

      LLVMRunFunctionPassManager(passmgr, func);
      func = LLVMGetNextFunction(func);
   }
   LLVMFinalizeFunctionPassManager(passmgr);
   if (gallivm_debug & GALLIVM_DEBUG_PERF) {
      int64_t time_end = os_time_get();
      int time_msec = (int)((time_end - time_begin) / 1000);
      
      const char *module_name = get_module_name(mod);
      debug_printf("optimizing module %s took %d msec\n",
                   module_name, time_msec);
   }

#if GALLIVM_HAVE_CORO == 1
   LLVMDisposePassManager(cgpassmgr);
#endif
   LLVMDisposePassManager(passmgr);
   return LLVMErrorSuccess;
#endif /* GALLIVM_USE_NEW_PASS */
}

LLVMErrorRef module_transform_wrapper(
      void *Ctx, LLVMOrcThreadSafeModuleRef *ModInOut,
      LLVMOrcMaterializationResponsibilityRef MR) {
   return LLVMOrcThreadSafeModuleWithModuleDo(*ModInOut, *module_transform, Ctx);
}

LPJit::LPJit() :jit_dylib_count(0) {
   using namespace llvm::orc;
#ifdef DEBUG
   ::gallivm_debug = debug_get_option_gallivm_debug();
#endif

   gallivm_perf = debug_get_flags_option("GALLIVM_PERF", lp_bld_perf_flags, 0 );

   init_native_targets();
   JITTargetMachineBuilder JTMB = create_jtdb();
   tm = wrap(ExitOnErr(JTMB.createTargetMachine()).release());

   /* Create an LLJIT instance with an ObjectLinkingLayer (JITLINK)
    * or RuntimeDyld as the base layer.
    * intel & perf listeners are not supported by ObjectLinkingLayer yet
    */
   lljit = ExitOnErr(
      LLJITBuilder()
         .setJITTargetMachineBuilder(std::move(JTMB))
#ifdef USE_JITLINK
         .setObjectLinkingLayerCreator(
            [&](ExecutionSession &ES, const llvm::Triple &TT) {
               return std::make_unique<ObjectLinkingLayer>(
                  ES, ExitOnErr(llvm::jitlink::InProcessMemoryManager::Create()));
            })
#else
#if LLVM_USE_INTEL_JITEVENTS
         .RegisterJITEventListener(
               llvm::JITEventListener::createIntelJITEventListener())
#endif
#endif
         .create());

   LLVMOrcIRTransformLayerRef TL = wrap(&lljit->getIRTransformLayer());
   LLVMOrcIRTransformLayerSetTransform(TL, *module_transform_wrapper, NULL);
}

void LPJit::init_native_targets() {
   // If we have a native target, initialize it to ensure it is linked in and
   // usable by the JIT.
   llvm::InitializeNativeTarget();

   llvm::InitializeNativeTargetAsmPrinter();

   llvm::InitializeNativeTargetDisassembler();
#if DEBUG
   {
      char *env_llc_options = getenv("GALLIVM_LLC_OPTIONS");
      if (env_llc_options) {
         char *option;
         char *options[64] = {(char *) "llc"};      // Warning without cast
         int   n;
         for (n = 0, option = strtok(env_llc_options, " "); option; n++, option = strtok(NULL, " ")) {
            options[n + 1] = option;
         }
         if (::gallivm_debug & (GALLIVM_DEBUG_IR | GALLIVM_DEBUG_ASM | GALLIVM_DEBUG_DUMP_BC)) {
            debug_printf("llc additional options (%d):\n", n);
            for (int i = 1; i <= n; i++)
               debug_printf("\t%s\n", options[i]);
            debug_printf("\n");
         }
         LLVMParseCommandLineOptions(n + 1, options, NULL);
      }
   }

   /* For simulating less capable machines */
   if (debug_get_bool_option("LP_FORCE_SSE2", false)) {
      extern struct util_cpu_caps_t util_cpu_caps;
      assert(util_cpu_caps.has_sse2);
      util_cpu_caps.has_sse3 = 0;
      util_cpu_caps.has_ssse3 = 0;
      util_cpu_caps.has_sse4_1 = 0;
      util_cpu_caps.has_sse4_2 = 0;
      util_cpu_caps.has_avx = 0;
      util_cpu_caps.has_avx2 = 0;
      util_cpu_caps.has_f16c = 0;
      util_cpu_caps.has_fma = 0;
   }
#endif

   if (util_get_cpu_caps()->has_avx2 || util_get_cpu_caps()->has_avx) {
      ::lp_native_vector_width = 256;
   } else {
      /* Leave it at 128, even when no SIMD extensions are available.
       * Really needs to be a multiple of 128 so can fit 4 floats.
       */
      ::lp_native_vector_width = 128;
   }

   ::lp_native_vector_width = debug_get_num_option("LP_NATIVE_VECTOR_WIDTH",
                                                 lp_native_vector_width);

#if LLVM_VERSION_MAJOR < 4
   if (::lp_native_vector_width <= 128) {
      /* Hide AVX support, as often LLVM AVX intrinsics are only guarded by
       * "util_get_cpu_caps()->has_avx" predicate, and lack the
       * "lp_native_vector_width > 128" predicate. And also to ensure a more
       * consistent behavior, allowing one to test SSE2 on AVX machines.
       * XXX: should not play games with util_cpu_caps directly as it might
       * get used for other things outside llvm too.
       */
      util_get_cpu_caps()->has_avx = 0;
      util_get_cpu_caps()->has_avx2 = 0;
      util_get_cpu_caps()->has_f16c = 0;
      util_get_cpu_caps()->has_fma = 0;
   }
#endif

#ifdef PIPE_ARCH_PPC_64
   /* Set the NJ bit in VSCR to 0 so denormalized values are handled as
    * specified by IEEE standard (PowerISA 2.06 - Section 6.3). This guarantees
    * that some rounding and half-float to float handling does not round
    * incorrectly to 0.
    * XXX: should eventually follow same logic on all platforms.
    * Right now denorms get explicitly disabled (but elsewhere) for x86,
    * whereas ppc64 explicitly enables them...
    */
   if (util_get_cpu_caps()->has_altivec) {
      unsigned short mask[] = { 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                                0xFFFF, 0xFFFF, 0xFFFE, 0xFFFF };
      __asm (
        "mfvscr %%v1\n"
        "vand   %0,%%v1,%0\n"
        "mtvscr %0"
        :
        : "r" (*mask)
      );
   }
#endif
}

llvm::orc::JITTargetMachineBuilder LPJit::create_jtdb() {
   using namespace llvm;
   using orc::JITTargetMachineBuilder;

#if defined(_WIN32) && LLVM_VERSION_MAJOR < 15
   /*
    * JITLink works on Windows, but only through ELF object format.
    *
    * XXX: We could use `LLVM_HOST_TRIPLE "-elf"` but LLVM_HOST_TRIPLE has
    * different strings for MinGW/MSVC, so better play it safe and be
    * explicit.
    */
#  ifdef _WIN64
   JITTargetMachineBuilder JTMB((Triple("x86_64-pc-win32-elf")));
#  else
   JITTargetMachineBuilder JTMB((Triple("i686-pc-win32-elf")));
#  endif
#else
   /*
    * llvm::sys::getProcessTriple() is bogus. It returns the host LLVM was 
    * compiled on. Be careful when doing cross compilation
    */
   JITTargetMachineBuilder JTMB((Triple(sys::getProcessTriple())));
#endif

   TargetOptions options;
   /**
    * LLVM 3.1+ haven't more "extern unsigned llvm::StackAlignmentOverride" and
    * friends for configuring code generation options, like stack alignment.
    */
#if defined(PIPE_ARCH_X86) && LLVM_VERSION_MAJOR < 13
   options.StackAlignmentOverride = 4;
#endif

#if defined(PIPE_ARCH_RISCV64)
#if defined(__riscv_float_abi_soft)
   options.MCOptions.ABIName = "lp64";
#elif defined(__riscv_float_abi_single)
   options.MCOptions.ABIName = "lp64f";
#elif defined(__riscv_float_abi_double)
   options.MCOptions.ABIName = "lp64d";
#else
#error "GALLIVM: unknown target riscv float abi"
#endif
#endif

#if defined(PIPE_ARCH_RISCV32)
#if defined(__riscv_float_abi_soft)
   options.MCOptions.ABIName = "ilp32";
#elif defined(__riscv_float_abi_single)
   options.MCOptions.ABIName = "ilp32f";
#elif defined(__riscv_float_abi_double)
   options.MCOptions.ABIName = "ilp32d";
#else
#error "GALLIVM: unknown target riscv float abi"
#endif
#endif

   JTMB.setOptions(options);

   std::vector<std::string> MAttrs;

#if LLVM_VERSION_MAJOR >= 4 && (defined(PIPE_ARCH_X86) || defined(PIPE_ARCH_X86_64) || defined(PIPE_ARCH_ARM))
   /* llvm-3.3+ implements sys::getHostCPUFeatures for Arm
    * and llvm-3.7+ for x86, which allows us to enable/disable
    * code generation based on the results of cpuid on these
    * architectures.
    */
   StringMap<bool> features;
   sys::getHostCPUFeatures(features);

   for (StringMapIterator<bool> f = features.begin();
        f != features.end();
        ++f) {
      MAttrs.push_back(((*f).second ? "+" : "-") + (*f).first().str());
   }
#elif defined(PIPE_ARCH_X86) || defined(PIPE_ARCH_X86_64)
   /*
    * We need to unset attributes because sometimes LLVM mistakenly assumes
    * certain features are present given the processor name.
    *
    * https://bugs.freedesktop.org/show_bug.cgi?id=92214
    * http://llvm.org/PR25021
    * http://llvm.org/PR19429
    * http://llvm.org/PR16721
    */
   MAttrs.push_back(util_get_cpu_caps()->has_sse    ? "+sse"    : "-sse"   );
   MAttrs.push_back(util_get_cpu_caps()->has_sse2   ? "+sse2"   : "-sse2"  );
   MAttrs.push_back(util_get_cpu_caps()->has_sse3   ? "+sse3"   : "-sse3"  );
   MAttrs.push_back(util_get_cpu_caps()->has_ssse3  ? "+ssse3"  : "-ssse3" );
   MAttrs.push_back(util_get_cpu_caps()->has_sse4_1 ? "+sse4.1" : "-sse4.1");
   MAttrs.push_back(util_get_cpu_caps()->has_sse4_2 ? "+sse4.2" : "-sse4.2");
   /*
    * AVX feature is not automatically detected from CPUID by the X86 target
    * yet, because the old (yet default) JIT engine is not capable of
    * emitting the opcodes. On newer llvm versions it is and at least some
    * versions (tested with 3.3) will emit avx opcodes without this anyway.
    */
   MAttrs.push_back(util_get_cpu_caps()->has_avx  ? "+avx"  : "-avx");
   MAttrs.push_back(util_get_cpu_caps()->has_f16c ? "+f16c" : "-f16c");
   MAttrs.push_back(util_get_cpu_caps()->has_fma  ? "+fma"  : "-fma");
   MAttrs.push_back(util_get_cpu_caps()->has_avx2 ? "+avx2" : "-avx2");
   /* disable avx512 and all subvariants */
   MAttrs.push_back("-avx512cd");
   MAttrs.push_back("-avx512er");
   MAttrs.push_back("-avx512f");
   MAttrs.push_back("-avx512pf");
   MAttrs.push_back("-avx512bw");
   MAttrs.push_back("-avx512dq");
   MAttrs.push_back("-avx512vl");
#endif
#if defined(PIPE_ARCH_ARM)
   if (!util_get_cpu_caps()->has_neon) {
      MAttrs.push_back("-neon");
      MAttrs.push_back("-crypto");
      MAttrs.push_back("-vfp2");
   }
#endif

#if defined(PIPE_ARCH_PPC)
   MAttrs.push_back(util_get_cpu_caps()->has_altivec ? "+altivec" : "-altivec");
#if (LLVM_VERSION_MAJOR < 4)
   /*
    * Make sure VSX instructions are disabled
    * See LLVM bugs:
    * https://llvm.org/bugs/show_bug.cgi?id=25503#c7 (fixed in 3.8.1)
    * https://llvm.org/bugs/show_bug.cgi?id=26775 (fixed in 3.8.1)
    * https://llvm.org/bugs/show_bug.cgi?id=33531 (fixed in 4.0)
    * https://llvm.org/bugs/show_bug.cgi?id=34647 (llc performance on certain unusual shader IR; intro'd in 4.0, pending as of 5.0)
    */
   if (util_get_cpu_caps()->has_altivec) {
      MAttrs.push_back("-vsx");
   }
#else
   /*
    * Bug 25503 is fixed, by the same fix that fixed
    * bug 26775, in versions of LLVM later than 3.8 (starting with 3.8.1).
    * BZ 33531 actually comprises more than one bug, all of
    * which are fixed in LLVM 4.0.
    *
    * With LLVM 4.0 or higher:
    * Make sure VSX instructions are ENABLED (if supported), unless
    * VSX instructions are explicitly enabled/disabled via GALLIVM_VSX=1 or 0.
    */
   if (util_get_cpu_caps()->has_altivec) {
      MAttrs.push_back(util_get_cpu_caps()->has_vsx ? "+vsx" : "-vsx");
   }
#endif
#endif

#if defined(PIPE_ARCH_MIPS64)
   MAttrs.push_back(util_get_cpu_caps()->has_msa ? "+msa" : "-msa");
   /* MSA requires a 64-bit FPU register file */
   MAttrs.push_back("+fp64");
#endif

#if defined(PIPE_ARCH_RISCV64)
   /* Before riscv is more matured and util_get_cpu_caps() is implemented,
    * assume this for now since most of linux capable riscv machine are
    * riscv64gc
    */
   MAttrs = {"+m","+c","+a","+d","+f"};
#endif

   JTMB.addFeatures(MAttrs);

   if (::gallivm_debug & (GALLIVM_DEBUG_IR | GALLIVM_DEBUG_ASM | GALLIVM_DEBUG_DUMP_BC)) {
      int n = MAttrs.size();
      if (n > 0) {
         debug_printf("llc -mattr option(s): ");
         for (int i = 0; i < n; i++)
            debug_printf("%s%s", MAttrs[i].c_str(), (i < n - 1) ? "," : "");
         debug_printf("\n");
      }
   }

   std::string MCPU = llvm::sys::getHostCPUName().str();
   /*
    * Note that the MAttrs set above will be sort of ignored (since we should
    * not set any which would not be set by specifying the cpu anyway).
    * It ought to be safe though since getHostCPUName() should include bits
    * not only from the cpu but environment as well (for instance if it's safe
    * to use avx instructions which need OS support). According to
    * http://llvm.org/bugs/show_bug.cgi?id=19429 however if I understand this
    * right it may be necessary to specify older cpu (or disable mattrs) though
    * when not using MCJIT so no instructions are generated which the old JIT
    * can't handle. Not entirely sure if we really need to do anything yet.
    * 
    * Not sure if the above is also the case for ORCJIT, but we need set CPU
    * manually since we don't use JITTargetMachineBuilder::detectHost()
    */

#ifdef PIPE_ARCH_PPC_64
   /*
    * Large programs, e.g. gnome-shell and firefox, may tax the addressability
    * of the Medium code model once dynamically generated JIT-compiled shader
    * programs are linked in and relocated.  Yet the default code model as of
    * LLVM 8 is Medium or even Small.
    * The cost of changing from Medium to Large is negligible:
    * - an additional 8-byte pointer stored immediately before the shader entrypoint;
    * - change an add-immediate (addis) instruction to a load (ld).
    */
   JTMB.setCodeModel(CodeModel::Large);

#if UTIL_ARCH_LITTLE_ENDIAN
   /*
    * Versions of LLVM prior to 4.0 lacked a table entry for "POWER8NVL",
    * resulting in (big-endian) "generic" being returned on
    * little-endian Power8NVL systems.  The result was that code that
    * attempted to load the least significant 32 bits of a 64-bit quantity
    * from memory loaded the wrong half.  This resulted in failures in some
    * Piglit tests, e.g.
    * .../arb_gpu_shader_fp64/execution/conversion/frag-conversion-explicit-double-uint
    */
   if (MCPU == "generic")
      MCPU = "pwr8";
#endif
#endif

#if defined(PIPE_ARCH_MIPS64)
   /*
    * ls3a4000 CPU and ls2k1000 SoC is a mips64r5 compatible with MSA SIMD
    * instruction set implemented, while ls3a3000 is mips64r2 compatible
    * only. getHostCPUName() return "generic" on all loongson
    * mips CPU currently. So we override the MCPU to mips64r5 if MSA is
    * implemented, feedback to mips64r2 for all other ordinary mips64 cpu.
    */
   if (MCPU == "generic")
      MCPU = util_get_cpu_caps()->has_msa ? "mips64r5" : "mips64r2";
#endif

#if defined(PIPE_ARCH_RISCV64)
   /**
    * should be fixed with https://reviews.llvm.org/D121149 in llvm 15,
    * set it anyway for llvm 14
    */
   if (MCPU == "generic")
      MCPU = "generic-rv64";

   JTMB.setCodeModel(CodeModel::Medium);
   JTMB.setRelocationModel(Reloc::PIC_);
#endif

#if defined(PIPE_ARCH_RISCV32)
   /**
    * should be fixed with https://reviews.llvm.org/D121149 in llvm 15,
    * set it anyway for llvm 14
    */
   if (MCPU == "generic")
      MCPU = "generic-rv32";

   JTMB.setCodeModel(CodeModel::Medium);
   JTMB.setRelocationModel(Reloc::PIC_);
#endif

   JTMB.setCPU(MCPU);
   if (gallivm_debug & (GALLIVM_DEBUG_IR | GALLIVM_DEBUG_ASM | GALLIVM_DEBUG_DUMP_BC)) {
      debug_printf("llc -mcpu option: %s\n", MCPU.c_str());
   }

   return JTMB;
}


} /* Anonymous namespace */

unsigned
lp_build_init_native_width(void)
{
   // Default to 256 until we're confident llvmpipe with 512 is as correct and not slower than 256
   lp_native_vector_width = MIN2(util_get_cpu_caps()->max_vector_bits, 256);
   assert(lp_native_vector_width);

   lp_native_vector_width = debug_get_num_option("LP_NATIVE_VECTOR_WIDTH", lp_native_vector_width);
   assert(lp_native_vector_width);

   return lp_native_vector_width;
}

bool
lp_build_init(void)
{
   (void)LPJit::get_instance();
   return true;
}

bool
init_gallivm_state(struct gallivm_state *gallivm, const char *name,
                   LLVMOrcThreadSafeContextRef context, struct lp_cached_code *cache)
{
   assert(!gallivm->context);
   assert(!gallivm->_ts_context);
   assert(!gallivm->module);

   if (!lp_build_init())
      return false;

   // cache is not implemented
   gallivm->cache = cache;
   if (gallivm->cache)
      gallivm->cache->data_size = 0;

   gallivm->_ts_context = context;
   gallivm->context = LLVMOrcThreadSafeContextGetContext(context);

   gallivm->module_name = LPJit::get_unique_name(name);
   gallivm->module = LLVMModuleCreateWithNameInContext(gallivm->module_name,
                                                       gallivm->context);
#if defined(PIPE_ARCH_X86)
   lp_set_module_stack_alignment_override(gallivm->module, 4);
#endif
   gallivm->builder = LLVMCreateBuilderInContext(gallivm->context);
   gallivm->_per_module_jd = LPJit::create_jit_dylib(gallivm->module_name);

   gallivm->target = LLVMCreateTargetDataLayout(LPJit::get_instance()->tm);

   return true;
}

struct gallivm_state *
gallivm_create(const char *name, LLVMOrcThreadSafeContextRef context,
               struct lp_cached_code *cache){
   struct gallivm_state *gallivm;

   gallivm = CALLOC_STRUCT(gallivm_state);
   if (gallivm) {
      if (!init_gallivm_state(gallivm, name, context, cache)) {
         FREE(gallivm);
         gallivm = NULL;
      }
   }

   assert(gallivm != NULL);
   return gallivm;
}

void
gallivm_destroy(struct gallivm_state *gallivm)
{
   LPJit::remove_jd(gallivm->_per_module_jd);
   gallivm->_per_module_jd = nullptr;
}

void
gallivm_free_ir(struct gallivm_state *gallivm)
{
   if (gallivm->module)
      LLVMDisposeModule(gallivm->module);
   FREE(gallivm->module_name);

   if (gallivm->target) {
      LLVMDisposeTargetData(gallivm->target);
   }

   if (gallivm->builder)
      LLVMDisposeBuilder(gallivm->builder);

   gallivm->target = NULL;
   gallivm->module=NULL;
   gallivm->module_name=NULL;
   gallivm->builder=NULL;
   gallivm->context=NULL;
   gallivm->_ts_context=NULL;
   gallivm->cache=NULL;
   LPJit::deregister_gallivm_state(gallivm);
}

void
gallivm_verify_function(struct gallivm_state *gallivm,
                        LLVMValueRef func)
{
   /* Verify the LLVM IR.  If invalid, dump and abort */
#ifdef DEBUG
   if (LLVMVerifyFunction(func, LLVMPrintMessageAction)) {
      lp_debug_dump_value(func);
      assert(0);
      return;
   }
#endif

   if (gallivm_debug & GALLIVM_DEBUG_IR) {
      /* Print the LLVM IR to stderr */
      lp_debug_dump_value(func);
      debug_printf("\n");
   }
return;
}

void
gallivm_add_global_mapping(struct gallivm_state *gallivm, LLVMValueRef sym, void* addr)
{
   LPJit::add_mapping_to_jd(sym, addr, gallivm->_per_module_jd);
}

void lp_init_clock_hook(struct gallivm_state *gallivm)
{
   if (gallivm->get_time_hook)
      return;

   LLVMTypeRef get_time_type = LLVMFunctionType(LLVMInt64TypeInContext(gallivm->context), NULL, 0, 1);
   gallivm->get_time_hook = LLVMAddFunction(gallivm->module, "get_time_hook", get_time_type);
}

void
gallivm_compile_module(struct gallivm_state *gallivm)
{
   if (gallivm->debug_printf_hook)
      gallivm_add_global_mapping(gallivm, gallivm->debug_printf_hook,
         (void *)debug_printf);

   LPJit::add_ir_module_to_jd(gallivm->_ts_context, gallivm->module,
      gallivm->_per_module_jd);
   /* ownership of module is now transferred into orc jit,
    * disallow modifying it
    */
   LPJit::register_gallivm_state(gallivm);
   gallivm->module=nullptr;

   /* defer compilation till first lookup by gallivm_jit_function */
}

func_pointer
gallivm_jit_function(struct gallivm_state *gallivm,
                     const char *func_name)
{
   return pointer_to_func(
      LPJit::lookup_in_jd(func_name, gallivm->_per_module_jd));
}

unsigned
gallivm_get_perf_flags(void){
   return gallivm_perf;
}

void
lp_set_module_stack_alignment_override(LLVMModuleRef MRef, unsigned align)
{
#if LLVM_VERSION_MAJOR >= 13
   llvm::Module *M = llvm::unwrap(MRef);
   M->setOverrideStackAlignment(align);
#endif
}
