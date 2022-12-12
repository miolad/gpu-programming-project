#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

namespace viewer {

extern "C" {

/// Initialize the viewer.
/// This function returns the library's context.
///
/// # Safety
///
/// `device_uuid` must point to a 16 byte buffer containing the UUID of the CUDA device
void *init(void *cuda_buffer_handle,
           uint64_t cuda_buffer_size,
           uint32_t res_x,
           uint32_t res_y,
           const uint8_t *device_uuid);

/// Run the event loop, rendering at most one frame, then returning control to the caller.
/// The return value is `true` if the application should close, `false` otherwise.
///
/// # Safety
///
/// `ctx` must be a pointer previously returned by [init], and not already deinitialized.
bool run_event_loop(void *ctx);

/// Deinitialize the library.
///
/// # Safety
///
/// `ctx` must be a pointer previously returned by [init], and not already deinitialized.
void deinit(void *ctx);

} // extern "C"

} // namespace viewer
