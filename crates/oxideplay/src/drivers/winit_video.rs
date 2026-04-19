//! wgpu-backed video renderer for the winit driver.
//!
//! Three `R8Unorm` textures for Y/U/V planes and a fragment shader
//! that does BT.709 YUV→RGB conversion. The Y texture is full-size;
//! U and V are half-size (4:2:0 chroma). The shader samples all three
//! with linear filtering, handling the upsample for free.

use std::sync::Arc;

use oxideav_core::{Error, Result, VideoFrame};
use crate::drivers::video_convert::to_yuv420p;

pub struct VideoRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_cfg: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// Adapter-reported maximum texture dimension. Surface width/height
    /// (and upload plane sizes) are clamped to this so that going
    /// fullscreen on a display larger than the GPU's limit — e.g.
    /// 4 K monitors with an adapter that reports 2048 — doesn't panic
    /// inside `surface.configure`.
    max_texture_dim: u32,
    /// The current (Y-plane, i.e. frame) dimensions. Textures are
    /// resized lazily when this changes.
    dims: Option<(u32, u32)>,
    textures: Option<YuvTextures>,
    bind_group: Option<wgpu::BindGroup>,
    /// Uniform buffer carrying the aspect-ratio letterbox scale + offset
    /// that the shader uses to decide where the content rectangle sits
    /// inside the surface.
    uniform_buffer: wgpu::Buffer,
    /// Whether we've already printed the "downscaling content" notice.
    /// One-shot so we don't spam the log on every frame of e.g. an
    /// 8 K source on a 4 K-limit adapter.
    warned_downscale: bool,
}

struct YuvTextures {
    y: wgpu::Texture,
    u: wgpu::Texture,
    v: wgpu::Texture,
}

impl VideoRenderer {
    /// Build a wgpu device + surface on `window`, configured to the
    /// window's inner size. `window` is held as an `Arc` so `wgpu` can
    /// own a `'static` surface without us giving up the handle.
    pub fn new(window: Arc<winit::window::Window>) -> Result<Self> {
        pollster::block_on(Self::new_async(window))
    }

    async fn new_async(window: Arc<winit::window::Window>) -> Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| Error::other(format!("wgpu: create_surface: {e}")))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| Error::other("wgpu: no suitable adapter"))?;

        // Use the adapter's native limits rather than the
        // `downlevel_defaults` preset — the latter caps max texture
        // dimension at 2048, which is smaller than any 4 K display
        // surface wants to be once the user clicks the fullscreen
        // button.
        let adapter_limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("oxideplay-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter_limits.clone(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| Error::other(format!("wgpu: request_device: {e}")))?;
        let max_texture_dim = adapter_limits.max_texture_dimension_2d;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| matches!(f, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm))
            .unwrap_or(caps.formats[0]);
        let surface_cfg = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.clamp(1, max_texture_dim),
            height: size.height.clamp(1, max_texture_dim),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &surface_cfg);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("yuv_to_rgb"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("yuv_to_rgb.wgsl").into(),
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("yuv-bgl"),
            entries: &[
                // Y
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // U
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // V
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // aspect-ratio uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("yuv-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("yuv-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("yuv-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // 16 bytes: one vec4<f32> aspect-ratio uniform.
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yuv-uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Default: no letterboxing (content fills the viewport). This
        // gets overwritten on the first render call once we know the
        // content dims.
        queue.write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&[1.0_f32, 1.0, 0.0, 0.0]));

        Ok(Self {
            device,
            queue,
            surface,
            surface_cfg,
            pipeline,
            bind_group_layout,
            sampler,
            max_texture_dim,
            dims: None,
            textures: None,
            bind_group: None,
            uniform_buffer,
            warned_downscale: false,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.surface_cfg.width = width.clamp(1, self.max_texture_dim);
        self.surface_cfg.height = height.clamp(1, self.max_texture_dim);
        self.surface.configure(&self.device, &self.surface_cfg);
    }

    pub fn render(&mut self, frame: &VideoFrame) -> Result<()> {
        let src_w = frame.width;
        let src_h = frame.height;
        if src_w == 0 || src_h == 0 {
            return Ok(());
        }

        // If the source exceeds the adapter's texture limit, fall back
        // to an integer-factor box downsample so we never hand wgpu a
        // dimension it can't honour. Keeps the full content visible at
        // reduced resolution instead of cropping or panicking.
        let (y_data, u_data, v_data, plane_w, plane_h) =
            prepare_planes(frame, self.max_texture_dim, &mut self.warned_downscale);
        if plane_w == 0 || plane_h == 0 {
            return Ok(());
        }

        if self.dims != Some((plane_w, plane_h)) {
            self.create_textures(plane_w, plane_h);
            self.dims = Some((plane_w, plane_h));
        }

        self.upload_plane(PlaneKind::Y, plane_w, plane_h, &y_data);
        self.upload_plane(PlaneKind::U, plane_w / 2, plane_h / 2, &u_data);
        self.upload_plane(PlaneKind::V, plane_w / 2, plane_h / 2, &v_data);

        // Update the letterbox uniform so the shader scales the content
        // rectangle to fit the surface while preserving the source
        // aspect ratio (pillar bars for wide content in a tall window,
        // letter bars for tall content in a wide window). We use the
        // SOURCE aspect (src_w / src_h) rather than the potentially
        // downsampled plane_w/plane_h so the content isn't squashed
        // when we've had to downsample to fit the texture limit.
        let surface_aspect = self.surface_cfg.width as f32 / self.surface_cfg.height.max(1) as f32;
        let content_aspect = src_w as f32 / src_h.max(1) as f32;
        let (sx, sy, ox, oy) = if content_aspect > surface_aspect {
            // Content is wider than surface — letterbox (black bars top/bottom).
            let h_frac = surface_aspect / content_aspect;
            let off_y = (1.0 - h_frac) * 0.5;
            (1.0, 1.0 / h_frac, 0.0, off_y)
        } else {
            // Content is taller than surface (or matches) — pillarbox.
            let w_frac = content_aspect / surface_aspect;
            let off_x = (1.0 - w_frac) * 0.5;
            (1.0 / w_frac, 1.0, off_x, 0.0)
        };
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[sx, sy, ox, oy]));

        let frame_tex = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                self.surface.configure(&self.device, &self.surface_cfg);
                self.surface.get_current_texture().map_err(|e| {
                    Error::other(format!("wgpu: reacquire surface texture: {e}"))
                })?
            }
            Err(e) => return Err(Error::other(format!("wgpu: surface texture: {e}"))),
        };
        let view = frame_tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("yuv-encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("yuv-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            if let Some(bg) = self.bind_group.as_ref() {
                pass.set_bind_group(0, bg, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        self.queue.submit(Some(encoder.finish()));
        frame_tex.present();
        Ok(())
    }

    fn create_textures(&mut self, w: u32, h: u32) {
        let y = self.make_plane_tex("y", w, h);
        let u = self.make_plane_tex("u", w / 2, h / 2);
        let v = self.make_plane_tex("v", w / 2, h / 2);
        let y_view = y.create_view(&wgpu::TextureViewDescriptor::default());
        let u_view = u.create_view(&wgpu::TextureViewDescriptor::default());
        let v_view = v.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("yuv-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&y_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&u_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&v_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });

        self.textures = Some(YuvTextures { y, u, v });
        self.bind_group = Some(bind_group);
        // Views are only referenced during bind-group creation; the
        // bind group holds its own references so we don't keep them.
        drop((y_view, u_view, v_view));
    }

    fn make_plane_tex(&self, label: &str, w: u32, h: u32) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    fn upload_plane(&self, kind: PlaneKind, w: u32, h: u32, data: &[u8]) {
        let Some(tex) = self.textures.as_ref() else {
            return;
        };
        let target = match kind {
            PlaneKind::Y => &tex.y,
            PlaneKind::U => &tex.u,
            PlaneKind::V => &tex.v,
        };
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(w),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
        );
    }
}

enum PlaneKind {
    Y,
    U,
    V,
}

/// Prepare YUV 4:2:0 planes sized to fit within `max_dim`. If the source
/// frame is larger than the limit, box-downsample all three planes by an
/// integer factor chosen so the largest dimension lands ≤ `max_dim`. The
/// output width/height are rounded down to even so the chroma
/// half-resolution math works.
fn prepare_planes(
    frame: &VideoFrame,
    max_dim: u32,
    warned: &mut bool,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, u32, u32) {
    let (mut y, mut u, mut v) = to_yuv420p(frame);
    let mut w = frame.width;
    let mut h = frame.height;
    // Y is w×h, U/V are (w/2)×(h/2). `max_dim` caps the Y plane.
    let longest = w.max(h);
    if longest <= max_dim {
        return (y, u, v, w, h);
    }
    // Smallest integer factor N such that ceil(longest / N) ≤ max_dim.
    let scale = longest.div_ceil(max_dim).max(2);
    let new_w = (w / scale) & !1;
    let new_h = (h / scale) & !1;
    if new_w == 0 || new_h == 0 {
        return (Vec::new(), Vec::new(), Vec::new(), 0, 0);
    }
    if !*warned {
        eprintln!(
            "oxideplay: source {}×{} exceeds GPU max texture dim {}; \
             downscaling to {}×{}",
            w, h, max_dim, new_w, new_h
        );
        *warned = true;
    }
    y = box_downsample(&y, w as usize, h as usize, scale as usize);
    u = box_downsample(&u, (w / 2) as usize, (h / 2) as usize, scale as usize);
    v = box_downsample(&v, (w / 2) as usize, (h / 2) as usize, scale as usize);
    w = new_w;
    h = new_h;
    (y, u, v, w, h)
}

/// Integer-factor box filter. Averages each `factor × factor` block of
/// the input plane into one output byte.
fn box_downsample(src: &[u8], src_w: usize, src_h: usize, factor: usize) -> Vec<u8> {
    if factor <= 1 {
        return src.to_vec();
    }
    let out_w = src_w / factor;
    let out_h = src_h / factor;
    let mut out = Vec::with_capacity(out_w * out_h);
    for oy in 0..out_h {
        for ox in 0..out_w {
            let mut acc = 0u32;
            for dy in 0..factor {
                for dx in 0..factor {
                    let sx = ox * factor + dx;
                    let sy = oy * factor + dy;
                    acc += src[sy * src_w + sx] as u32;
                }
            }
            out.push((acc / (factor * factor) as u32) as u8);
        }
    }
    out
}
