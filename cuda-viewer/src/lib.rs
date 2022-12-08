mod viewer;

use core::ffi::c_void;
use viewer::Viewer;
use winit::event_loop::EventLoop;

struct Context {
    viewer: Viewer,
    event_loop: EventLoop<()>
}

/// Initialize the viewer.
/// This function returns the library's context.
/// 
/// # Safety
/// 
/// `device_uuid` must point to a 16 byte buffer containing the UUID of the CUDA device
#[no_mangle]
pub unsafe extern "C" fn init(
    cuda_buffer_handle: *mut c_void,
    cuda_buffer_size: u64,
    res_x: u32,
    res_y: u32,
    device_uuid: *const u8
) -> *mut c_void {
    let device_uuid = std::slice::from_raw_parts(device_uuid, 16);
    let event_loop = EventLoop::new();
    let viewer = Viewer::new(&event_loop, cuda_buffer_handle, cuda_buffer_size, res_x, res_y, device_uuid);
    
    let ctx = Box::new(Context {
        event_loop,
        viewer
    });

    Box::into_raw(ctx) as _
}

/// Run the event loop, rendering at most one frame, then returning control to the caller.
/// The return value is `true` if the application should close, `false` otherwise.
/// 
/// # Safety
/// 
/// `ctx` must be a pointer previously returned by [init], and not already deinitialized.
#[no_mangle]
pub unsafe extern "C" fn run_event_loop(ctx: *mut c_void) -> bool {
    let mut ctx = Box::from_raw(ctx as *mut Context);

    let should_close = ctx.viewer.run_return(&mut ctx.event_loop);

    Box::leak(ctx);
    should_close
}

/// Deinitialize the library.
/// 
/// # Safety
/// 
/// `ctx` must be a pointer previously returned by [init], and not already deinitialized.
#[no_mangle]
pub unsafe extern "C" fn deinit(ctx: *mut c_void) {
    let _ = Box::from_raw(ctx as *mut Context);
}
