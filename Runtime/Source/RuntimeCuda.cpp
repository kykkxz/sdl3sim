#include "RuntimeCuda.hpp"

namespace sim::runtime {

#if SIM_ENABLE_CUDA
extern "C" int SimCudaProbeKernel();
extern "C" int SimCudaRunBallPhysics(
    CudaParticleState* particles,
    int count,
    const CudaPhysicsParams* params,
    CudaPhysicsDiagnostics* diagnostics
);
#endif

bool IsCudaBuildEnabled() {
#if SIM_ENABLE_CUDA
    return true;
#else
    return false;
#endif
}

bool IsCudaRuntimeAvailable() {
#if SIM_ENABLE_CUDA
    return SimCudaProbeKernel() != 0;
#else
    return false;
#endif
}

bool RunCudaBallPhysics(
    std::vector<CudaParticleState>& particles,
    const CudaPhysicsParams& params,
    CudaPhysicsDiagnostics& outDiagnostics
) {
#if SIM_ENABLE_CUDA
    if (particles.empty()) {
        outDiagnostics = CudaPhysicsDiagnostics{};
        return true;
    }
    const int result = SimCudaRunBallPhysics(
        particles.data(),
        static_cast<int>(particles.size()),
        &params,
        &outDiagnostics
    );
    return result != 0;
#else
    (void)particles;
    (void)params;
    outDiagnostics = CudaPhysicsDiagnostics{};
    return false;
#endif
}

}  // namespace sim::runtime
