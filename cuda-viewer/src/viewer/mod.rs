mod shaders;

#[cfg(feature = "shaderc")]
use shaderc::ShaderKind;
#[cfg(feature = "shaderc")]
use shaders::ShaderCompiler;
#[cfg(feature = "shaderc")]
use colored::{Color, Colorize};
use shaders::create_vk_shader_module;

use core::ffi::{CStr, c_char};
use std::{os::raw::c_void, collections::HashSet};
use ash::{vk::{self, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT, DebugUtilsMessengerCallbackDataEXT, Bool32}, Entry, extensions::{ext::DebugUtils, khr::{Surface, Swapchain}}, Instance, Device};
use winit::{window::{WindowBuilder, Window}, dpi::PhysicalSize, event_loop::{EventLoop, ControlFlow}, event::{Event, WindowEvent, KeyboardInput, VirtualKeyCode, ElementState}, platform::run_return::EventLoopExtRunReturn};
use raw_window_handle::{HasRawWindowHandle, HasRawDisplayHandle};

const USE_VALIDATION: bool = cfg!(debug_assertions);
const VALIDATION_LAYERS: [&CStr; 1] = unsafe {[
    CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0")
]};

#[cfg(all(not(feature = "zero-copy-mem"), target_os = "windows"))]
const EXTERNAL_MEMORY_EXT_NAME: &CStr = ash::extensions::khr::ExternalMemoryWin32::name();
#[cfg(all(not(feature = "zero-copy-mem"), target_os = "linux"))]
const EXTERNAL_MEMORY_EXT_NAME: &CStr = ash::extensions::khr::ExternalMemoryFd::name();

// Measures how tightly CPU and GPU should be synchronized together
const FRAMES_IN_FLIGHT: usize = 2;

unsafe extern "system" fn debug_callback(
    message_severity: DebugUtilsMessageSeverityFlagsEXT,
    _message_types: DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void
) -> Bool32 {
    #[cfg(feature = "shaderc")]
    let color = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => Color::Red,
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => Color::Yellow,
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => Color::Blue,
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => Color::Green,

        _ => unreachable!()
    };
    
    if message_severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        #[cfg(feature = "shaderc")]
        println!("{}: {}", "Validation layer".color(color), CStr::from_ptr((*p_callback_data).p_message).to_string_lossy());
        #[cfg(not(feature = "shaderc"))]
        println!("Validation layer: {}", CStr::from_ptr((*p_callback_data).p_message).to_string_lossy());
    }

    vk::FALSE
}

pub struct Viewer {
    window: Window,
    debug_utils_messenger: Option<vk::DebugUtilsMessengerEXT>,
    surface: vk::SurfaceKHR,
    queue: vk::Queue,
    present_queue: vk::Queue,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<(vk::Image, vk::ImageView)>,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    swapchain_framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Box<[vk::CommandBuffer; FRAMES_IN_FLIGHT]>,
    swapchain_extent: vk::Extent2D,
    render_done_fences: Box<[vk::Fence; FRAMES_IN_FLIGHT]>,
    image_acquired_semaphores: Box<[vk::Semaphore; FRAMES_IN_FLIGHT]>,
    render_done_semaphores: Box<[vk::Semaphore; FRAMES_IN_FLIGHT]>,
    cuda_buffer: (vk::Buffer, vk::DeviceMemory, u64),

    /// Used by the frames in flight logic to decide which set of objects to use on which frame
    frame_index: usize,

    // These objects are all loaders, i.e. contain function pointers.
    // They have to be dropped in the opposite order they are created, thus their order here is
    // important, since Rust drops fields of a struct from top to bottom
    swapchain_loader: Swapchain,
    device: Box<Device>,
    surface_loader: Surface,
    debug_loader: Option<DebugUtils>,
    instance: Box<Instance>,
    _entry: Box<Entry>
}

type SyncObjects = (Box<[vk::Fence; FRAMES_IN_FLIGHT]>, Box<[vk::Semaphore; FRAMES_IN_FLIGHT]>, Box<[vk::Semaphore; FRAMES_IN_FLIGHT]>);

impl Viewer {
    pub fn new(event_loop: &EventLoop<()>, cuda_buffer_handle: *mut c_void, cuda_buffer_size: u64, res_x: u32, res_y: u32, device_uuid: &[u8]) -> Self {
        unsafe {
            let entry = Box::new(Entry::load().expect("Vulkan is not available"));
            
            let window = Self::init_window(event_loop, res_x, res_y);
            Self::init_vulkan(entry, window, cuda_buffer_handle, cuda_buffer_size, device_uuid)
        }
    }
    
    pub fn run_return(&mut self, event_loop: &mut EventLoop<()>) -> bool {
        unsafe {
            self.main_loop(event_loop)
        }
    }
}

impl Viewer {
    unsafe fn init_vulkan(entry: Box<Entry>, window: Window, cuda_buffer_handle: *mut c_void, cuda_buffer_size: u64, device_uuid: &[u8]) -> Self {
        let use_validation = if USE_VALIDATION {
            Self::check_validation_layer_support(&entry)
        } else {
            false
        };
        let instance = Box::new(Self::create_instance(&entry, &window, use_validation));
        let (debug_loader, debug_utils_messenger) = Self::setup_debug_messenger(&entry, &instance, use_validation);
        let surface_loader = Surface::new(&entry, &instance);
        let surface = ash_window::create_surface(&entry, &instance, window.raw_display_handle(), window.raw_window_handle(), None).expect("Couldn't create surface");
        let (physical_device, device, queue, present_queue, graphics_queue_family, present_queue_family) = 
            Self::pick_device(&instance, &surface_loader, &surface, device_uuid, use_validation);
        let (swapchain_loader, swapchain, format, swapchain_extent) =
            Self::create_swapchain(&instance, &device, &window, &surface_loader, surface, physical_device, From::from([graphics_queue_family, present_queue_family]));
        let swapchain_images = Self::get_swapchain_images(&device, &swapchain_loader, swapchain, format);
        let render_pass = Self::create_render_pass(&device, format);
        let (pipeline_layout, pipeline) = Self::create_graphics_pipeline(&device, render_pass);
        let swapchain_framebuffers =
            Self::create_swapchain_framebuffers(&device, render_pass, swapchain_extent, swapchain_images.iter().map(|(_, view)| *view));
        let command_pool = Self::create_command_pool(&device, graphics_queue_family);
        let command_buffers = Self::allocate_command_buffers(&device, command_pool);
        let (render_done_fences, image_acquired_semaphores, render_done_semaphores) =
            Self::create_sync_objects(&device);
        let cuda_buffer = {
            #[cfg(feature = "zero-copy-mem")]
            let (buf, mem) = import_host_buffer(
                &instance,
                &device,
                physical_device,
                cuda_buffer_handle,
                cuda_buffer_size,
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk::MemoryPropertyFlags::empty() // Allow any possible memory property
            ).unwrap();

            #[cfg(not(feature = "zero-copy-mem"))]
            let (buf, mem) = import_external_buffer(
                &instance,
                &device,
                physical_device,
                cuda_buffer_handle,
                cuda_buffer_size,
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk::MemoryPropertyFlags::DEVICE_LOCAL
            ).unwrap();

            let buffer_device_address_info = vk::BufferDeviceAddressInfo::builder()
                .buffer(buf);
            let dev_addr = device.get_buffer_device_address(&buffer_device_address_info);

            (buf, mem, dev_addr)
        };

        Self {
            _entry: entry,
            instance,
            window,
            debug_utils_messenger,
            debug_loader,
            surface_loader,
            surface,
            device,
            queue,
            present_queue,
            swapchain_loader,
            swapchain,
            swapchain_images,
            render_pass,
            pipeline_layout,
            pipeline,
            swapchain_framebuffers,
            command_pool,
            command_buffers,
            swapchain_extent,
            render_done_fences,
            image_acquired_semaphores,
            render_done_semaphores,
            frame_index: 0,
            cuda_buffer
        }
    }

    unsafe fn main_loop(&mut self, event_loop: &mut EventLoop<()>) -> bool {
        static mut IS_MINIMIZED: bool = false;
        let mut should_close = false;
        
        event_loop.run_return(|event, _, control_flow| {
            match event {
                Event::WindowEvent {event, ..} => {
                    match event {
                        WindowEvent::CloseRequested => {
                            should_close = true;
                            *control_flow = ControlFlow::Exit;
                        },

                        WindowEvent::KeyboardInput {
                            input: KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                state: ElementState::Pressed,
                                ..
                            },
                            ..
                        } => {
                            should_close = true;
                            *control_flow = ControlFlow::Exit;
                        },

                        WindowEvent::Resized(PhysicalSize { width, height }) => {
                            IS_MINIMIZED = width != self.swapchain_extent.width && height != self.swapchain_extent.height;
                        },

                        _ => {}
                    }
                }

                Event::MainEventsCleared => self.window.request_redraw(),

                Event::RedrawRequested(_) => {
                    // Draw one frame, then return
                    if !IS_MINIMIZED {
                        self.draw_frame();
                    }
                    *control_flow = ControlFlow::Exit;
                },

                Event::LoopDestroyed => self.device.device_wait_idle().unwrap(),

                _ => {}
            }
        });

        should_close
    }

    unsafe fn draw_frame(&mut self) {
        // Choose which set of objects to use for this frame
        let frame_index = self.frame_index % FRAMES_IN_FLIGHT;
        self.frame_index = (self.frame_index + 1) % FRAMES_IN_FLIGHT;

        // Acquire swapchain image
        let image_index = self.swapchain_loader.acquire_next_image(
            self.swapchain,
            std::u64::MAX,
            self.image_acquired_semaphores[frame_index],
            vk::Fence::null()
        ).unwrap().0;

        // Before recording the command buffer, make sure that the previous frame has finished by waiting on a fence
        self.device.wait_for_fences(std::slice::from_ref(&self.render_done_fences[frame_index]), true, std::u64::MAX).unwrap();

        // Reset the fence
        self.device.reset_fences(std::slice::from_ref(&self.render_done_fences[frame_index])).unwrap();

        // Record this frame's command buffer
        self.record_command_buffer(self.command_buffers[frame_index], image_index as _);

        // Submit the command buffer making sure to properly sync with the `image_acquired_semaphore`
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(std::slice::from_ref(&self.image_acquired_semaphores[frame_index]))
            .wait_dst_stage_mask(std::slice::from_ref(&vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT))
            .command_buffers(std::slice::from_ref(&self.command_buffers[frame_index]))
            .signal_semaphores(std::slice::from_ref(&self.render_done_semaphores[frame_index]));

        self.device.queue_submit(self.queue, std::slice::from_ref(&submit_info), self.render_done_fences[frame_index]).unwrap();

        // Present the image to the screen after it's done rendering
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(std::slice::from_ref(&self.render_done_semaphores[frame_index]))
            .swapchains(std::slice::from_ref(&self.swapchain))
            .image_indices(std::slice::from_ref(&image_index));
        
        self.swapchain_loader.queue_present(self.present_queue, &present_info).unwrap();
    }

    unsafe fn record_command_buffer(&self, command_buffer: vk::CommandBuffer, image_index: usize) {
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        self.device.begin_command_buffer(command_buffer, &command_buffer_begin_info).unwrap();

        let clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0]
            }
        };

        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(self.swapchain_framebuffers[image_index])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D {x: 0, y: 0},
                extent: self.swapchain_extent
            })
            .clear_values(std::slice::from_ref(&clear_value));

        self.device.cmd_begin_render_pass(command_buffer, &render_pass_begin_info, vk::SubpassContents::INLINE);
        self.device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
        self.device.cmd_push_constants(command_buffer, self.pipeline_layout, vk::ShaderStageFlags::FRAGMENT, 0, &self.cuda_buffer.2.to_ne_bytes());
        self.device.cmd_push_constants(command_buffer, self.pipeline_layout, vk::ShaderStageFlags::FRAGMENT, 8, &self.swapchain_extent.width.to_ne_bytes());

        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(self.swapchain_extent.width as _)
            .height(self.swapchain_extent.height as _)
            .min_depth(0.0)
            .max_depth(0.0);

        self.device.cmd_set_viewport(command_buffer, 0, std::slice::from_ref(&viewport));

        let scissor = vk::Rect2D {
            offset: vk::Offset2D {
                x: 0,
                y: 0
            },
            extent: self.swapchain_extent
        };

        self.device.cmd_set_scissor(command_buffer, 0, std::slice::from_ref(&scissor));
        self.device.cmd_draw(command_buffer, 6, 1, 0, 0);
        self.device.cmd_end_render_pass(command_buffer);
        self.device.end_command_buffer(command_buffer).unwrap();
    }

    unsafe fn create_sync_objects(device: &Device) -> SyncObjects {
        let fence_create_info = vk::FenceCreateInfo::builder()
            .flags(vk::FenceCreateFlags::SIGNALED);
        
        let semaphore_create_info = vk::SemaphoreCreateInfo::default();

        let (fences, (semaphores1, semaphores2)): (Vec<_>, (Vec<_>, Vec<_>)) = (0..FRAMES_IN_FLIGHT).into_iter()
            .map(|_| {
                let fence = device.create_fence(&fence_create_info, None).unwrap();
                let semaphore1 = device.create_semaphore(&semaphore_create_info, None).unwrap();
                let semaphore2 = device.create_semaphore(&semaphore_create_info, None).unwrap();

                (fence, (semaphore1, semaphore2))
            })
            .unzip();
        
        (
            fences.into_boxed_slice().try_into().unwrap(),
            semaphores1.into_boxed_slice().try_into().unwrap(),
            semaphores2.into_boxed_slice().try_into().unwrap()
        )
    }

    unsafe fn allocate_command_buffers(device: &Device, command_pool: vk::CommandPool) -> Box<[vk::CommandBuffer; FRAMES_IN_FLIGHT]> {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(2);

        device.allocate_command_buffers(&allocate_info)
            .unwrap()
            .into_boxed_slice()
            .try_into()
            .unwrap()
    }

    unsafe fn create_command_pool(device: &Device, graphics_queue_family: u32) -> vk::CommandPool {
        let create_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(graphics_queue_family);

        device.create_command_pool(&create_info, None).unwrap()
    }

    unsafe fn create_swapchain_framebuffers<V: Iterator<Item = vk::ImageView>>(
        device: &Device,
        render_pass: vk::RenderPass,
        views_extent: vk::Extent2D,
        image_views: V
    ) -> Vec<vk::Framebuffer>
    {
        image_views
            .map(|view| {
                let create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass)
                    .attachments(std::slice::from_ref(&view))
                    .layers(1)
                    .width(views_extent.width)
                    .height(views_extent.height);

                device.create_framebuffer(&create_info, None).unwrap()
            })
            .collect()
    }

    unsafe fn create_render_pass(device: &Device, format: vk::Format) -> vk::RenderPass {
        let attachment = vk::AttachmentDescription::builder()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let attachment_reference = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            
        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&attachment_reference));

        let entry_dependency = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::NONE)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .build();

        // let exit_dependency = vk::SubpassDependency::builder()
        //     .src_subpass(0)
        //     .dst_subpass(vk::SUBPASS_EXTERNAL)
        //     .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        //     .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        //     .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        //     .dst_access_mask(vk::AccessFlags::NONE)
        //     .build();
        
        // let dependencies = [entry_dependency, exit_dependency];
            
        let render_pass_create_info = vk::RenderPassCreateInfo::builder()
            .attachments(std::slice::from_ref(&attachment))
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(std::slice::from_ref(&entry_dependency));

        device.create_render_pass(&render_pass_create_info, None).unwrap()
    }

    unsafe fn create_graphics_pipeline(device: &Device, render_pass: vk::RenderPass) -> (vk::PipelineLayout, vk::Pipeline) {
        #[cfg(feature = "shaderc")]
        let shader_compiler = ShaderCompiler::new().unwrap();
        #[cfg(feature = "shaderc")]
        let vertex_shader_module = {
            let (binary, warnings) = shader_compiler.compile(
                "fs_quad.vert",
                ShaderKind::Vertex,
                "main"
            ).unwrap();
            // println!("fs_quad.vert: {}", warnings);

            create_vk_shader_module(device, &binary).unwrap()
        };
        #[cfg(feature = "shaderc")]
        let fragment_shader_module = {
            let (binary, warnings) = shader_compiler.compile(
                "buf_display.frag",
                ShaderKind::Fragment,
                "main"
            ).unwrap();
            // println!("buf_display.frag: {}", warnings);

            create_vk_shader_module(device, &binary).unwrap()
        };

        #[cfg(not(feature = "shaderc"))]
        let vertex_shader_module = {
            let binary = std::include_bytes!("../shaders/fs_quad.vert.spv");
            let binary = unsafe {
                std::slice::from_raw_parts(binary.as_ptr() as *const u32, binary.len() / 4)
            };
            create_vk_shader_module(device, binary).unwrap()
        };
        #[cfg(not(feature = "shaderc"))]
        let fragment_shader_module = {
            let binary = std::include_bytes!("../shaders/buf_display.frag.spv");
            let binary = unsafe {
                std::slice::from_raw_parts(binary.as_ptr() as *const u32, binary.len() / 4)
            };
            create_vk_shader_module(device, binary).unwrap()
        };

        let vertex_shader_stage = vk::PipelineShaderStageCreateInfo::builder()
            .module(vertex_shader_module)
            .stage(vk::ShaderStageFlags::VERTEX)
            .name(CStr::from_bytes_with_nul_unchecked(b"main\0"));

        let fragment_shader_stage = vk::PipelineShaderStageCreateInfo::builder()
            .module(fragment_shader_module)
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .name(CStr::from_bytes_with_nul_unchecked(b"main\0"));

        let push_constant_range = vk::PushConstantRange::builder()
            .offset(0)
            .size(12)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_create_info, None).unwrap();

        let color_blend_attachment_state = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);
        
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .attachments(std::slice::from_ref(&color_blend_attachment_state))
            .logic_op_enable(false);

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(false)
            .stencil_test_enable(false);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&dynamic_states);

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .primitive_restart_enable(false)
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_shading_enable(false);

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false)
            .line_width(1.0);

        let stages = [vertex_shader_stage.build(), fragment_shader_stage.build()];

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default();

        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewport_count(1)
            .scissor_count(1);
        
        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
            .color_blend_state(&color_blend_state)
            .depth_stencil_state(&depth_stencil_state)
            .dynamic_state(&dynamic_state)
            .input_assembly_state(&input_assembly_state)
            .layout(pipeline_layout)
            .multisample_state(&multisample_state)
            .rasterization_state(&rasterization_state)
            .stages(&stages)
            .vertex_input_state(&vertex_input_state)
            .viewport_state(&viewport_state)
            .render_pass(render_pass)
            .subpass(0);

        let pipeline = device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_create_info),
            None
        ).unwrap()[0];

        // Destroy shader modules
        device.destroy_shader_module(vertex_shader_module, None);
        device.destroy_shader_module(fragment_shader_module, None);

        (pipeline_layout, pipeline)
    }

    fn create_debug_utils_debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT {
        vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::INFO |
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING |
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
            .message_type(vk::DebugUtilsMessageTypeFlagsEXT::GENERAL |
                vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION |
                vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE)
            .pfn_user_callback(Some(debug_callback))
            .build()
    }

    unsafe fn get_swapchain_images(device: &Device, swapchain_loader: &Swapchain, swapchain: vk::SwapchainKHR, format: vk::Format) -> Vec<(vk::Image, vk::ImageView)> {
        swapchain_loader.get_swapchain_images(swapchain)
            .unwrap()
            .into_iter()
            .map(|image| {
                let view_create_info = vk::ImageViewCreateInfo::builder()
                    .components(vk::ComponentMapping::builder()
                        .r(vk::ComponentSwizzle::IDENTITY)
                        .g(vk::ComponentSwizzle::IDENTITY)
                        .b(vk::ComponentSwizzle::IDENTITY)
                        .a(vk::ComponentSwizzle::IDENTITY)
                        .build()
                    )
                    .format(format)
                    .image(image)
                    .subresource_range(vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_array_layer(0)
                        .base_mip_level(0)
                        .layer_count(1)
                        .level_count(1)
                        .build()
                    )
                    .view_type(vk::ImageViewType::TYPE_2D);
                
                let image_view = device.create_image_view(&view_create_info, None).unwrap();

                (image, image_view)
            })
            .collect()
    }

    unsafe fn create_swapchain(
            instance: &Instance,
            device: &Device,
            window: &Window,
            surface_loader: &Surface,
            surface: vk::SurfaceKHR,
            physical_device: vk::PhysicalDevice,
            queue_families_set: HashSet<u32>
        ) -> (Swapchain, vk::SwapchainKHR, vk::Format, vk::Extent2D)
    {
        let swapchain_loader = Swapchain::new(instance, device);

        // Pick a suitable format and color space
        let (format, color_space) = {
            let formats = surface_loader.get_physical_device_surface_formats(physical_device, surface)
                .unwrap();

            formats
                .iter()
                .map(|format| (format.format, format.color_space))
                .find(|(format, color_space)| {
                    *format == vk::Format::B8G8R8A8_SRGB && *color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
                .unwrap_or((formats[0].format, formats[0].color_space))
        };

        let surface_capabilities = surface_loader.get_physical_device_surface_capabilities(physical_device, surface)
            .unwrap();

        let swapchain_extent = match surface_capabilities.current_extent.width {
            std::u32::MAX => {
                let size = window.inner_size();
                vk::Extent2D {
                    width: size.width.clamp(surface_capabilities.min_image_extent.width, surface_capabilities.max_image_extent.width),
                    height: size.height.clamp(surface_capabilities.min_image_extent.height, surface_capabilities.max_image_extent.height)
                }
            },
            _ => surface_capabilities.current_extent
        };

        let requested_image_count = (surface_capabilities.min_image_count + 1)
            .min(if surface_capabilities.max_image_count > 0 { surface_capabilities.max_image_count } else { std::u32::MAX });
        
        let queue_families: Vec<u32> = queue_families_set.into_iter().collect();
        
        let mut swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .clipped(true)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .image_array_layers(1)
            .image_color_space(color_space)
            .image_extent(swapchain_extent)
            .image_format(format)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .min_image_count(requested_image_count)
            .pre_transform(surface_capabilities.current_transform)
            .present_mode(vk::PresentModeKHR::IMMEDIATE)
            .surface(surface)
            .queue_family_indices(&queue_families);
        
        swapchain_create_info.image_sharing_mode = if queue_families.len() > 1 {
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };

        let swapchain = swapchain_loader.create_swapchain(&swapchain_create_info, None)
            .expect("Couldn't create the swapchain");

        (swapchain_loader, swapchain, format, swapchain_extent)
    }

    unsafe fn pick_device(
        instance: &Instance,
        surface_loader: &Surface,
        surface: &vk::SurfaceKHR,
        physical_device_uuid: &[u8],
        use_validation: bool
    ) -> (vk::PhysicalDevice, Box<Device>, vk::Queue, vk::Queue, u32, u32)
    {
        let ret = instance.enumerate_physical_devices()
            .unwrap()
            .iter()
            .find_map(|&physical_device| {
                // Get physical device properties
                let mut id_properties = vk::PhysicalDeviceIDProperties::default();
                let mut physical_device_properties = vk::PhysicalDeviceProperties2::builder()
                    .push_next(&mut id_properties)
                    .build();
                instance.get_physical_device_properties2(physical_device, &mut physical_device_properties);
                
                // Check UUID match
                if id_properties.device_uuid != physical_device_uuid {
                    return None;
                }
                
                // Check API version support
                let api_version = physical_device_properties.properties.api_version;
                let api_version_sufficient = match vk::api_version_major(api_version) {
                    0 => false,
                    1 => vk::api_version_minor(api_version) >= 2,
                    2.. => true,
                };
                if !api_version_sufficient {
                    return None;
                }

                // Check swapchain extension support
                let swapchain_ext_supported = instance.enumerate_device_extension_properties(physical_device)
                    .unwrap()
                    .into_iter()
                    .any(|ext| CStr::from_ptr(ext.extension_name.as_ptr()) == Swapchain::name());

                if !swapchain_ext_supported {
                    return None;
                }

                // Check external memory extension support
                #[cfg(feature = "zero-copy-mem")]
                let external_memory_ext_name = vk::ExtExternalMemoryHostFn::name();
                #[cfg(not(feature = "zero-copy-mem"))]
                let external_memory_ext_name = EXTERNAL_MEMORY_EXT_NAME;
                let external_memory_supported = instance.enumerate_device_extension_properties(physical_device)
                    .unwrap()
                    .into_iter()
                    .any(|ext| CStr::from_ptr(ext.extension_name.as_ptr()) == external_memory_ext_name);

                if !external_memory_supported {
                    return None;
                }

                // Check if there is at least one supported surface format
                let surface_format_supported = !surface_loader.get_physical_device_surface_formats(physical_device, *surface)
                    .unwrap()
                    .is_empty();

                if !surface_format_supported {
                    return None;
                }
                
                let mut graphics_queue = None;
                let mut present_queue = None;
                
                for (index, queue_family_props) in instance.get_physical_device_queue_family_properties(physical_device).into_iter().enumerate() {
                    if queue_family_props.queue_flags.contains(vk::QueueFlags::GRAPHICS) && graphics_queue.is_none() {
                        graphics_queue = Some(index as u32);
                    }

                    if surface_loader.get_physical_device_surface_support(physical_device, index as _, *surface).unwrap() && present_queue.is_none() {
                        present_queue = Some(index as u32);
                    }

                    if graphics_queue.is_some() && present_queue.is_some() {
                        break;
                    }
                }

                if let Some(graphics_queue_family) = graphics_queue {
                    if let Some(present_queue_family) = present_queue {
                        let prio = [1.0];
                        let queue_create_infos = HashSet::from([graphics_queue_family, present_queue_family])
                            .iter()
                            .map(|&queue_family| {
                                vk::DeviceQueueCreateInfo::builder()
                                    .queue_family_index(queue_family)
                                    .queue_priorities(&prio)
                                    .build()
                            })
                            .collect::<Vec<_>>();

                        let device_features = vk::PhysicalDeviceFeatures::builder()
                            .shader_int64(true)
                            .build();
                        let mut vk12_device_features = vk::PhysicalDeviceVulkan12Features::builder()
                            .buffer_device_address(true);
                        let mut extended_features = vk::PhysicalDeviceFeatures2::builder()
                            .features(device_features)
                            .push_next(&mut vk12_device_features);
                        
                        let mut layers = vec![];
                        
                        if use_validation {
                            layers.extend(VALIDATION_LAYERS.iter().map(|l| l.as_ptr()));
                        }

                        let enabled_extensions = [
                            Swapchain::name().as_ptr(),
                            #[cfg(feature = "zero-copy-mem")]
                            vk::ExtExternalMemoryHostFn::name().as_ptr(),
                            #[cfg(not(feature = "zero-copy-mem"))]
                            EXTERNAL_MEMORY_EXT_NAME.as_ptr()
                        ];

                        let device_create_info = vk::DeviceCreateInfo::builder()
                            .enabled_extension_names(&enabled_extensions)
                            .queue_create_infos(&queue_create_infos)
                            .enabled_layer_names(&layers)
                            .push_next(&mut extended_features);

                        let device = Box::new(instance.create_device(physical_device, &device_create_info, None)
                            .expect("Couldn't create logical device"));

                        let graphics_queue = device.get_device_queue(graphics_queue_family, 0);
                        let present_queue = device.get_device_queue(present_queue_family, 0);

                        Some((
                            physical_device,
                            device,
                            graphics_queue,
                            present_queue,
                            graphics_queue_family,
                            present_queue_family
                        ))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .expect("No physical device that meets the requirements is present");

        // println!("Selected device: {}", CStr::from_ptr(instance.get_physical_device_properties(ret.0).device_name.as_ptr()).to_string_lossy());

        ret
    }

    unsafe fn create_instance(entry: &Entry, window: &Window, use_validation: bool) -> ash::Instance {
        let app_info = vk::ApplicationInfo::builder()
            .application_name(CStr::from_bytes_with_nul_unchecked(b"CUDA PathTracing Viewer\0"))
            .application_version(1)
            .engine_name(CStr::from_bytes_with_nul_unchecked(b"No Engine\0"))
            .engine_version(1)
            .api_version(vk::API_VERSION_1_2);

        let extension_names = Self::get_required_extensions(window, use_validation);
        let mut layer_names = vec![];
            
        if use_validation {
            layer_names.extend(VALIDATION_LAYERS.iter().map(|&layer| layer.as_ptr()));
        }

        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .enabled_layer_names(&layer_names);

        let mut debug_messenger_create_info = Self::create_debug_utils_debug_messenger_create_info();
            
        let create_info = if use_validation {
            create_info.push_next(&mut debug_messenger_create_info)
        } else {
            create_info
        };

        entry.create_instance(&create_info, None).expect("Failed to create instance!")
    }

    unsafe fn check_validation_layer_support(entry: &Entry) -> bool {
        let layers = entry.enumerate_instance_layer_properties()
            .expect("Couldn't enumerate instance layer properties")
            .into_iter()
            .map(|prop| CStr::from_ptr(prop.layer_name.as_ptr()).to_owned())
            .collect::<Vec<_>>();

        VALIDATION_LAYERS
            .iter()
            .all(|&l| layers.contains(&l.to_owned()))
    }

    unsafe fn get_required_extensions(window: &Window, use_validation: bool) -> Vec<*const c_char> {
        let mut extension_names = ash_window::enumerate_required_extensions(window.raw_display_handle())
            .unwrap()
            .to_vec();

        if use_validation {
            extension_names.push(DebugUtils::name().as_ptr());
        }

        extension_names
    }

    unsafe fn setup_debug_messenger(entry: &Entry, instance: &Instance, use_validation: bool) -> (Option<DebugUtils>, Option<vk::DebugUtilsMessengerEXT>) {
        if !use_validation {
            (None, None)
        } else {
            let debug_utils = DebugUtils::new(entry, instance);

            let create_info = Self::create_debug_utils_debug_messenger_create_info();
            let messenger = debug_utils.create_debug_utils_messenger(&create_info, None)
                .expect("Couldn't create debug utils messenger");

            (Some(debug_utils), Some(messenger))
        }
    }

    fn init_window(event_loop: &EventLoop<()>, res_x: u32, res_y: u32) -> Window {
        WindowBuilder::new()
            .with_title("CUDA Viewer")
            .with_inner_size(PhysicalSize::new(res_x, res_y))
            .with_resizable(false)
            .build(event_loop)
            .expect("Unable to build winit Window")
    }
}

impl Drop for Viewer {
    fn drop(&mut self) {
        unsafe {
            if let Some(debug_loader) = self.debug_loader.take() {
                if let Some(messenger) = self.debug_utils_messenger {
                    debug_loader.destroy_debug_utils_messenger(messenger, None);
                }
            }

            self.device.destroy_buffer(self.cuda_buffer.0, None);
            self.device.free_memory(self.cuda_buffer.1, None);
            
            for (fence, (sem1, sem2)) in self.render_done_fences.iter().zip(self.image_acquired_semaphores.iter().zip(self.render_done_semaphores.iter())) {
                self.device.destroy_fence(*fence, None);
                self.device.destroy_semaphore(*sem1, None);
                self.device.destroy_semaphore(*sem2, None);
            }

            self.device.destroy_command_pool(self.command_pool, None);

            for fb in &self.swapchain_framebuffers {
                self.device.destroy_framebuffer(*fb, None);
            }

            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);

            for (_, view) in &self.swapchain_images {
                self.device.destroy_image_view(*view, None);
            }

            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

unsafe fn find_memory_type(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    requirements: u32,
    desired_properties: vk::MemoryPropertyFlags
) -> Option<u32>
{
    let available_types = instance.get_physical_device_memory_properties(physical_device).memory_types;

    available_types.into_iter()
        .enumerate()
        .map(|(i, typ)| (i as u32, typ.property_flags))
        .find_map(|(i, flags)| {
            let type_compatible = (requirements & (1 << i)) != 0;
            let has_desired_properties = (flags & desired_properties) == desired_properties;

            (type_compatible && has_desired_properties).then_some(i)
        })
}

/// Imports a buffer from an external resource, like a CUDA context.
/// 
/// Can return None if no available memory type is compatible
/// In case this function returns None, it is possible the memory and/or buffer have been allocated anyway
#[cfg(not(feature = "zero-copy-mem"))]
unsafe fn import_external_buffer(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    handle: *mut c_void,
    size: u64,
    usage: vk::BufferUsageFlags,
    memory_properties: vk::MemoryPropertyFlags
) -> Option<(vk::Buffer, vk::DeviceMemory)> {
    let mut external_memory_buffer_info = vk::ExternalMemoryBufferCreateInfo::builder()
        .handle_types(if cfg!(target_os = "windows") { vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32 } else { vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD });
    
    let buffer_create_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut external_memory_buffer_info);

    let buffer = device.create_buffer(&buffer_create_info, None).ok()?;
    let memory_requirements = device.get_buffer_memory_requirements(buffer);

    #[cfg(target_os = "windows")]
    let mut handle_info = vk::ImportMemoryWin32HandleInfoKHR::builder()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32)
        .handle(handle)
        .name(std::ptr::null_mut());
    #[cfg(target_os = "linux")]
    let mut handle_info = vk::ImportMemoryFdInfoKHR::builder()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD)
        .fd(handle as _);
    
    let mut allocate_flags_info = vk::MemoryAllocateFlagsInfo::builder()
        .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let allocate_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(memory_requirements.size)
        .memory_type_index(find_memory_type(instance, physical_device, memory_requirements.memory_type_bits, memory_properties)?)
        .push_next(&mut allocate_flags_info)
        .push_next(&mut handle_info);

    let allocation = device.allocate_memory(&allocate_info, None).ok()?;
    device.bind_buffer_memory(buffer, allocation, 0).ok()?;
    
    Some((buffer, allocation))
}

/// Imports a buffer from an host allocated memory
/// 
/// Can return None if no available memory type is compatible
/// In case this function returns None, it is possible the memory and/or buffer have been allocated anyway
#[cfg(feature = "zero-copy-mem")]
unsafe fn import_host_buffer(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    host_buf: *mut c_void,
    size: u64,
    usage: vk::BufferUsageFlags,
    memory_properties: vk::MemoryPropertyFlags
) -> Option<(vk::Buffer, vk::DeviceMemory)> {
    let mut external_memory_buffer_info = vk::ExternalMemoryBufferCreateInfo::builder()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
    
    let buffer_create_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut external_memory_buffer_info);

    let buffer = device.create_buffer(&buffer_create_info, None).ok()?;
    let memory_requirements = device.get_buffer_memory_requirements(buffer);

    let mut host_ptr_properties = vk::MemoryHostPointerPropertiesEXT::default();
    let external_memory_host_fn = vk::ExtExternalMemoryHostFn::load(|name| {
        std::mem::transmute(instance.get_device_proc_addr(device.handle(), name.as_ptr()))
    });
    (external_memory_host_fn.get_memory_host_pointer_properties_ext)(
        device.handle(),
        vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
        host_buf as _,
        &mut host_ptr_properties
    ).result().unwrap();

    if memory_requirements.memory_type_bits & host_ptr_properties.memory_type_bits == 0 {
        panic!("External memory host: no suitable memory types");
    }

    let mut import_memory_host_pointer_info = vk::ImportMemoryHostPointerInfoEXT::builder()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT)
        .host_pointer(host_buf);

    let mut allocate_flags_info = vk::MemoryAllocateFlagsInfo::builder()
        .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let allocate_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(memory_requirements.size)
        .memory_type_index(find_memory_type(instance, physical_device, memory_requirements.memory_type_bits & host_ptr_properties.memory_type_bits, memory_properties)?)
        .push_next(&mut allocate_flags_info)
        .push_next(&mut import_memory_host_pointer_info);

    let allocation = device.allocate_memory(&allocate_info, None).ok()?;
    device.bind_buffer_memory(buffer, allocation, 0).ok()?;
    
    Some((buffer, allocation))
}
