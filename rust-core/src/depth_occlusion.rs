use glow::HasContext;

/// Unpacks an ARCore `DEPTH16` pixel stored as `[low_byte, high_byte]` in `GL_RG8` to millimeters.
pub fn unpack_rg8_mm(low: u8, high: u8) -> f32 {
    (low as f32) + (high as f32) * 256.0
}

/// Reads the center pixel distance (in meters) from tightly packed `DEPTH16` bytes.
pub fn center_depth_meters(bytes: &[u8], width: i32, height: i32) -> f32 {
    if width <= 0 || height <= 0 {
        return 0.0;
    }
    let cx = (width / 2) as usize;
    let cy = (height / 2) as usize;
    let idx = (cy * (width as usize) + cx) * 2;
    if idx + 1 >= bytes.len() {
        return 0.0;
    }
    let mm = unpack_rg8_mm(bytes[idx], bytes[idx + 1]);
    if mm <= 1.0 {
        0.0
    } else {
        mm * 0.001
    }
}

/// CPU-staged depth frame received from ARCore `acquireDepthImage16Bits()`.
#[derive(Default)]
pub struct PendingDepthFrame {
    pub width: i32,
    pub height: i32,
    pub data: Vec<u8>,
    pub d0: [f32; 2],
    pub dx: [f32; 2],
    pub dy: [f32; 2],
    pub generation: u64,
    pub center_depth_m: f32,
}

impl PendingDepthFrame {
    pub fn update_from_slice(
        &mut self,
        width: i32,
        height: i32,
        bytes: &[u8],
        d0: [f32; 2],
        dx: [f32; 2],
        dy: [f32; 2],
    ) {
        let needed = (width.max(0) as usize) * (height.max(0) as usize) * 2;
        if needed == 0 || bytes.len() < needed {
            return;
        }
        self.width = width;
        self.height = height;
        self.data.resize(needed, 0);
        self.data.copy_from_slice(&bytes[..needed]);
        self.d0 = d0;
        self.dx = dx;
        self.dy = dy;
        self.center_depth_m = center_depth_meters(&self.data, width, height);
        self.generation = self.generation.wrapping_add(1).max(1);
    }
}

/// GPU-side `GL_RG8` depth texture uploaded on the GL thread when generation advances.
pub struct DepthTextureGpu {
    pub texture: glow::Texture,
    tex_w: i32,
    tex_h: i32,
    uploaded_gen: u64,
    pub has_valid_depth: bool,
    pub d0: [f32; 2],
    pub dx: [f32; 2],
    pub dy: [f32; 2],
}

impl DepthTextureGpu {
    pub fn new(gl: &glow::Context) -> Result<Self, String> {
        unsafe {
            let texture = gl.create_texture()?;
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );
            let placeholder = [0u8, 0u8];
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 2);
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RG8 as i32,
                1,
                1,
                0,
                glow::RG,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&placeholder)),
            );
            gl.bind_texture(glow::TEXTURE_2D, None);

            Ok(Self {
                texture,
                tex_w: 1,
                tex_h: 1,
                uploaded_gen: 0,
                has_valid_depth: false,
                d0: [0.0, 0.0],
                dx: [1.0, 0.0],
                dy: [0.0, 1.0],
            })
        }
    }

    pub fn upload_if_needed(&mut self, gl: &glow::Context, pending: &PendingDepthFrame) {
        if pending.generation == 0 || pending.generation == self.uploaded_gen {
            return;
        }
        let needed = (pending.width.max(0) as usize) * (pending.height.max(0) as usize) * 2;
        if needed == 0 || pending.data.len() < needed {
            return;
        }

        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(self.texture));
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 2);
            if self.tex_w != pending.width || self.tex_h != pending.height {
                gl.tex_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    glow::RG8 as i32,
                    pending.width,
                    pending.height,
                    0,
                    glow::RG,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(Some(&pending.data[..needed])),
                );
                self.tex_w = pending.width;
                self.tex_h = pending.height;
            } else {
                gl.tex_sub_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    0,
                    0,
                    pending.width,
                    pending.height,
                    glow::RG,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(Some(&pending.data[..needed])),
                );
            }
            gl.bind_texture(glow::TEXTURE_2D, None);
        }

        self.uploaded_gen = pending.generation;
        self.has_valid_depth = true;
        self.d0 = pending.d0;
        self.dx = pending.dx;
        self.dy = pending.dy;
    }

    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_texture(self.texture);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpacks_16bit_depth_little_endian_accurately() {
        // 1250 = 4 * 256 + 226
        assert!((unpack_rg8_mm(226, 4) - 1250.0).abs() < 1e-4);
    }

    #[test]
    fn extracts_center_depth_in_meters() {
        let mut buf = vec![0u8; 4 * 4 * 2];
        // Center (2, 2) => pixel index 10 => byte offset 20..21 => 2000mm (0x07D0 => low=0xD0=208, high=0x07=7)
        buf[20] = 0xD0;
        buf[21] = 0x07;
        assert!((center_depth_meters(&buf, 4, 4) - 2.0).abs() < 1e-4);
    }
}
