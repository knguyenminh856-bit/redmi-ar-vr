use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};
use std::time::Instant;

pub const MAX_RING_SLOTS: usize = 4;
pub const MAX_Y_BYTES: usize = 1280 * 768;
pub const MAX_UV_BYTES: usize = 1280 * 384;

const LUT_Y_BINS: usize = 128;
const LUT_UV_BINS: usize = 64;
const YUV_LUT_LEN: usize = LUT_Y_BINS * LUT_UV_BINS * LUT_UV_BINS; // 524,288 * 4B = 2.0 MB
const SKIN_LUT_LEN: usize = LUT_UV_BINS * LUT_UV_BINS; // 4,096 B = 4 KB
const MOTION_GRID_W: usize = 32;
const MOTION_GRID_H: usize = 24;
const MOTION_GRID_LEN: usize = MOTION_GRID_W * MOTION_GRID_H;

#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct CaptureMeta {
    pub seq: u64,
    pub cam_pos: [f32; 3],
    pub cam_rot: [f32; 4],
    pub tan_h: f32,
    pub tan_v: f32,
    pub ndc_basis: [f32; 6],
    pub rot_deg: i32,
}

struct YuvSlot {
    meta: CaptureMeta,
    width: usize,
    height: usize,
    y_len: usize,
    u_len: usize,
    v_len: usize,
    y_row_stride: usize,
    y_pixel_stride: usize,
    uv_row_stride: usize,
    uv_pixel_stride: usize,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

impl Default for YuvSlot {
    fn default() -> Self {
        Self {
            meta: CaptureMeta::default(),
            width: 0,
            height: 0,
            y_len: 0,
            u_len: 0,
            v_len: 0,
            y_row_stride: 0,
            y_pixel_stride: 1,
            uv_row_stride: 0,
            uv_pixel_stride: 1,
            y: vec![0u8; MAX_Y_BYTES],
            u: vec![0u8; MAX_UV_BYTES],
            v: vec![0u8; MAX_UV_BYTES],
        }
    }
}

pub struct VisionRamRing {
    slots: [RwLock<YuvSlot>; MAX_RING_SLOTS],
    latest_slot: AtomicUsize,
    latest_seq: AtomicU64,
    prev_luma_grid: Mutex<[u8; MOTION_GRID_LEN]>,
}

impl Default for VisionRamRing {
    fn default() -> Self {
        Self {
            slots: [
                RwLock::new(YuvSlot::default()),
                RwLock::new(YuvSlot::default()),
                RwLock::new(YuvSlot::default()),
                RwLock::new(YuvSlot::default()),
            ],
            latest_slot: AtomicUsize::new(0),
            latest_seq: AtomicU64::new(0),
            prev_luma_grid: Mutex::new([0u8; MOTION_GRID_LEN]),
        }
    }
}

static YUV_LUT: OnceLock<Box<[u32; YUV_LUT_LEN]>> = OnceLock::new();
static SKIN_LUT: OnceLock<[u8; SKIN_LUT_LEN]> = OnceLock::new();
static VISION_RING: OnceLock<VisionRamRing> = OnceLock::new();

fn build_yuv_lut() -> Box<[u32; YUV_LUT_LEN]> {
    let mut table = vec![0u32; YUV_LUT_LEN].into_boxed_slice();
    for yi in 0..LUT_Y_BINS {
        let y_full = ((yi << 1) | 1) as i32;
        let y = y_full - 16;
        let c = if y > 0 { 298 * y } else { 0 };
        let y_off = yi << 12;
        for ui in 0..LUT_UV_BINS {
            let u_full = ((ui << 2) + 2) as i32;
            let u = u_full - 128;
            let uv_off = y_off | (ui << 6);
            for vi in 0..LUT_UV_BINS {
                let v_full = ((vi << 2) + 2) as i32;
                let v = v_full - 128;

                let r = ((c + 409 * v + 128) >> 8).clamp(0, 255) as u8;
                let g = ((c - 100 * u - 208 * v + 128) >> 8).clamp(0, 255) as u8;
                let b = ((c + 516 * u + 128) >> 8).clamp(0, 255) as u8;
                table[uv_off | vi] = u32::from_le_bytes([r, g, b, 255]);
            }
        }
    }
    table
        .try_into()
        .unwrap_or_else(|_| unreachable!("YUV_LUT length verified"))
}

fn build_skin_lut() -> [u8; SKIN_LUT_LEN] {
    let mut mask = [0u8; SKIN_LUT_LEN];
    for ui in 0..LUT_UV_BINS {
        let u_full = (ui << 2) + 2;
        for vi in 0..LUT_UV_BINS {
            let v_full = (vi << 2) + 2;
            if (77..=127).contains(&u_full) && (133..=175).contains(&v_full) {
                mask[(ui << 6) | vi] = 1;
            }
        }
    }
    mask
}

fn yuv_lut() -> &'static [u32; YUV_LUT_LEN] {
    YUV_LUT.get_or_init(build_yuv_lut)
}

fn skin_lut() -> &'static [u8; SKIN_LUT_LEN] {
    SKIN_LUT.get_or_init(build_skin_lut)
}

fn ring() -> &'static VisionRamRing {
    VISION_RING.get_or_init(VisionRamRing::default)
}

/// Pre-allocates the 2MB YUV->RGBA LUT, 4KB skin mask, and 4-slot camera frame ring in RAM.
pub fn init_vision_ram() {
    let _ = yuv_lut();
    let _ = skin_lut();
    let _ = ring();
}

/// Copies incoming camera YUV_420_888 planes into a lock-free ring slot in <0.1ms so callers
/// can close `android.media.Image` immediately without blocking the GL render thread.
#[allow(clippy::too_many_arguments)]
pub fn ingest_camera_planes(
    y_src: &[u8],
    u_src: &[u8],
    v_src: &[u8],
    width: usize,
    height: usize,
    y_row_stride: usize,
    y_pixel_stride: usize,
    uv_row_stride: usize,
    uv_pixel_stride: usize,
    mut meta: CaptureMeta,
) -> u64 {
    if width < 2 || height < 2 || y_src.is_empty() || u_src.is_empty() || v_src.is_empty() {
        return 0;
    }
    let r = ring();
    let start_slot = (r.latest_slot.load(Ordering::Relaxed) + 1) % MAX_RING_SLOTS;

    for attempt in 0..MAX_RING_SLOTS {
        let slot_idx = (start_slot + attempt) % MAX_RING_SLOTS;
        let Ok(mut slot) = r.slots[slot_idx].try_write() else {
            continue;
        };

        let y_len = y_src.len();
        let u_len = u_src.len();
        let v_len = v_src.len();
        if slot.y.len() < y_len {
            slot.y.resize(y_len, 0);
        }
        if slot.u.len() < u_len {
            slot.u.resize(u_len, 0);
        }
        if slot.v.len() < v_len {
            slot.v.resize(v_len, 0);
        }

        slot.y[..y_len].copy_from_slice(y_src);
        slot.u[..u_len].copy_from_slice(u_src);
        slot.v[..v_len].copy_from_slice(v_src);

        slot.width = width;
        slot.height = height;
        slot.y_len = y_len;
        slot.u_len = u_len;
        slot.v_len = v_len;
        slot.y_row_stride = y_row_stride.max(width);
        slot.y_pixel_stride = y_pixel_stride.max(1);
        slot.uv_row_stride = uv_row_stride.max(width / 2);
        slot.uv_pixel_stride = uv_pixel_stride.max(1);

        let seq = r.latest_seq.fetch_add(1, Ordering::AcqRel) + 1;
        meta.seq = seq;
        slot.meta = meta;
        drop(slot);

        r.latest_slot.store(slot_idx, Ordering::Release);
        return seq;
    }

    0
}

/// Converts the latest cached camera slot directly to upright (`0°`) tightly-packed RGBA
/// using the 2MB 3D LUT + 2x2 luma averaging, returning `(meta, skin_permille, motion_permille, convert_us)`.
pub fn convert_latest_rgba(
    out_rgba: &mut [u8],
    dst_w: usize,
    dst_h: usize,
    last_seq: u64,
    update_motion: bool,
) -> Option<(CaptureMeta, u32, u32, u32)> {
    if dst_w < 2 || dst_h < 2 || out_rgba.len() < dst_w * dst_h * 4 {
        return None;
    }
    let r = ring();
    let newest_seq = r.latest_seq.load(Ordering::Acquire);
    if newest_seq == 0 || newest_seq == last_seq {
        return None;
    }

    let slot_idx = r.latest_slot.load(Ordering::Acquire) % MAX_RING_SLOTS;
    let slot = r.slots[slot_idx].read().ok()?;
    if slot.meta.seq == 0 || slot.meta.seq == last_seq || slot.width < 2 || slot.height < 2 {
        return None;
    }

    let t0 = Instant::now();
    let lut = yuv_lut();
    let skin = skin_lut();
    let src_w = slot.width;
    let src_h = slot.height;
    let y_len = slot.y_len;
    let u_len = slot.u_len;
    let v_len = slot.v_len;
    let y_row_stride = slot.y_row_stride;
    let y_pixel_stride = slot.y_pixel_stride;
    let uv_row_stride = slot.uv_row_stride;
    let uv_pixel_stride = slot.uv_pixel_stride;
    let rot180 = slot.meta.rot_deg == 180;
    let total_px = dst_w * dst_h;

    let mut skin_hits: u32 = 0;
    let mut skin_samples: u32 = 0;

    for dy in 0..dst_h {
        let sy0 = (dy * src_h) / dst_h;
        let sy1 = (sy0 + 1).min(src_h - 1);
        let y_row0 = sy0 * y_row_stride;
        let y_row1 = sy1 * y_row_stride;
        let uv_row = (sy0 >> 1) * uv_row_stride;
        let row_base = dy * dst_w;

        for dx in 0..dst_w {
            let sx0 = (dx * src_w) / dst_w;
            let sx1 = (sx0 + 1).min(src_w - 1);
            let y00_idx = y_row0 + sx0 * y_pixel_stride;
            let y01_idx = y_row0 + sx1 * y_pixel_stride;
            let y10_idx = y_row1 + sx0 * y_pixel_stride;
            let y11_idx = y_row1 + sx1 * y_pixel_stride;

            let y00 = if y00_idx < y_len { slot.y[y00_idx] as u32 } else { 0 };
            let y01 = if y01_idx < y_len { slot.y[y01_idx] as u32 } else { y00 };
            let y10 = if y10_idx < y_len { slot.y[y10_idx] as u32 } else { y00 };
            let y11 = if y11_idx < y_len { slot.y[y11_idx] as u32 } else { y00 };
            let y_avg = (y00 + y01 + y10 + y11) >> 2;

            let uv_idx = uv_row + (sx0 >> 1) * uv_pixel_stride;
            let u_raw = if uv_idx < u_len { slot.u[uv_idx] } else { 128 };
            let v_raw = if uv_idx < v_len { slot.v[uv_idx] } else { 128 };

            let yi = (y_avg >> 1).min((LUT_Y_BINS - 1) as u32) as usize;
            let ui = (u_raw >> 2) as usize;
            let vi = (v_raw >> 2) as usize;
            let rgba_bytes = lut[(yi << 12) | (ui << 6) | vi].to_le_bytes();

            let out_px = if rot180 {
                total_px - 1 - (row_base + dx)
            } else {
                row_base + dx
            };
            let out_off = out_px * 4;
            out_rgba[out_off..out_off + 4].copy_from_slice(&rgba_bytes);

            if y_avg > 45 {
                skin_samples += 1;
                skin_hits += skin[(ui << 6) | vi] as u32;
            }
        }
    }

    let motion_permille = if update_motion {
        if let Ok(mut prev) = r.prev_luma_grid.lock() {
            let mut moved: u32 = 0;
            for gy in 0..MOTION_GRID_H {
                let sy = ((gy * 2 + 1) * src_h) / (MOTION_GRID_H * 2);
                let row_off = sy * y_row_stride;
                for gx in 0..MOTION_GRID_W {
                    let sx = ((gx * 2 + 1) * src_w) / (MOTION_GRID_W * 2);
                    let idx = row_off + sx * y_pixel_stride;
                    let sample = if idx < y_len { slot.y[idx] } else { 0 };
                    let g_idx = gy * MOTION_GRID_W + gx;
                    if (sample as i32 - prev[g_idx] as i32).abs() >= 14 {
                        moved += 1;
                    }
                    prev[g_idx] = sample;
                }
            }
            (moved * 1000) / (MOTION_GRID_LEN as u32)
        } else {
            0
        }
    } else {
        0
    };

    let skin_permille = if skin_samples > 0 {
        (skin_hits * 1000) / skin_samples
    } else {
        0
    };
    let convert_us = t0.elapsed().as_micros().clamp(1, u32::MAX as u128) as u32;
    Some((slot.meta, skin_permille, motion_permille, convert_us))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yuv_lut_converts_white_black_and_skin_mask_cleanly() {
        init_vision_ram();
        let lut = yuv_lut();
        let skin = skin_lut();

        // Neutral bright Y=235, U=128, V=128 -> near white
        let white_idx = ((235usize >> 1) << 12) | ((128usize >> 2) << 6) | (128usize >> 2);
        let [wr, wg, wb, wa] = lut[white_idx].to_le_bytes();
        assert!(wr > 240 && wg > 240 && wb > 240 && wa == 255);

        // Dark Y=16, U=128, V=128 -> near black
        let black_idx = ((16usize >> 1) << 12) | ((128usize >> 2) << 6) | (128usize >> 2);
        let [br, bg, bb, ba] = lut[black_idx].to_le_bytes();
        assert!(br < 15 && bg < 15 && bb < 15 && ba == 255);

        // Typical skin chroma (Cb=105, Cr=150) should be flagged
        assert_eq!(skin[((105usize >> 2) << 6) | (150usize >> 2)], 1);
        // Cool cyan chroma should not be flagged as skin
        assert_eq!(skin[((180usize >> 2) << 6) | (60usize >> 2)], 0);
    }

    #[test]
    fn ring_buffer_ingests_rotates_180_and_skips_duplicate_seq() {
        init_vision_ram();
        let w = 8usize;
        let h = 4usize;
        // Left half bright (Y=220, skin chroma), right half dark (Y=20, neutral chroma)
        let mut y = vec![20u8; w * h];
        for row in 0..h {
            for col in 0..4 {
                y[row * w + col] = 220;
            }
        }
        let u = vec![106u8; (w / 2) * (h / 2)];
        let v = vec![150u8; (w / 2) * (h / 2)];

        let seq = ingest_camera_planes(
            &y,
            &u,
            &v,
            w,
            h,
            w,
            1,
            w / 2,
            1,
            CaptureMeta {
                rot_deg: 180,
                tan_h: 0.72,
                tan_v: 0.54,
                ..CaptureMeta::default()
            },
        );
        assert!(seq > 0);

        let mut rgba = vec![0u8; 4 * 2 * 4];
        let res = convert_latest_rgba(&mut rgba, 4, 2, 0, true);
        let (meta, skin_permille, _motion, _us) = res.expect("frame should convert");
        assert_eq!(meta.seq, seq);
        assert!(skin_permille > 400);

        // With 180 deg rotation, bright left pixels move to bottom-right / first pixels become dark
        let first_px_r = rgba[0];
        let last_px_r = rgba[(4 * 2 - 1) * 4];
        assert!(first_px_r < 55 && last_px_r > 230);

        // Duplicate sequence returns None cleanly
        assert!(convert_latest_rgba(&mut rgba, 4, 2, seq, false).is_none());
    }
}
