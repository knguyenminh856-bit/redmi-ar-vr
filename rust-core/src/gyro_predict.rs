use glam::Quat;
use std::f32::consts::{FRAC_PI_2, PI};

/// Maps Android `Surface.ROTATION_*` (0, 1, 2, 3) to a rotation from
/// the display-aligned head basis into the natural device sensor basis.
///
/// Android defines `ROTATION_90` as rotating the physical device 90 deg CCW
/// (so UI/rendering is rotated 90 deg CW relative to natural device axes).
pub fn display_to_device_quat(display_rotation: i32) -> Quat {
    match display_rotation {
        1 => Quat::from_rotation_z(-FRAC_PI_2), // Surface.ROTATION_90
        2 => Quat::from_rotation_z(PI),         // Surface.ROTATION_180
        3 => Quat::from_rotation_z(FRAC_PI_2),  // Surface.ROTATION_270
        _ => Quat::IDENTITY,                    // Surface.ROTATION_0
    }
}

/// Converts a raw quaternion from Android `TYPE_GAME_ROTATION_VECTOR`
/// (`Z-up` sensor world, natural device axes) into OpenGL Right-Handed
/// `Y-up` world coordinates (`+X` right, `+Y` up, `-Z` forward), aligned
/// with the active display orientation.
pub fn sensor_to_head_quat(raw_sensor_q: Quat, display_rotation: i32) -> Quat {
    let c_sensor_to_gl = Quat::from_rotation_x(-FRAC_PI_2);
    let q = c_sensor_to_gl * raw_sensor_q * display_to_device_quat(display_rotation);
    if q.length_squared() > 1e-8 {
        q.normalize()
    } else {
        Quat::IDENTITY
    }
}

/// Computes the local-frame rotation delta between capture time and latest 120Hz sample.
pub fn compute_local_delta(q_at_capture: Quat, q_latest: Quat) -> Quat {
    let delta = q_at_capture.inverse() * q_latest;
    if delta.length_squared() > 1e-8 {
        delta.normalize()
    } else {
        Quat::IDENTITY
    }
}

/// Composes the latest 120Hz gyro rotational delta onto the last 30Hz ARCore camera pose,
/// eliminating rotational lag between camera frames regardless of world yaw offset.
pub fn predict_ar_rotation(
    ar_cam_rot: Quat,
    gyro_at_ar_capture: Quat,
    gyro_latest: Quat,
) -> Quat {
    let predicted = ar_cam_rot * compute_local_delta(gyro_at_ar_capture, gyro_latest);
    if predicted.length_squared() > 1e-8 {
        predicted.normalize()
    } else {
        ar_cam_rot
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn assert_vec_close(a: Vec3, b: Vec3) {
        assert!(
            (a - b).length() < 1e-4,
            "expected {:?} to be close to {:?}",
            a,
            b
        );
    }

    #[test]
    fn sensor_world_basis_maps_to_opengl_yup() {
        let c = Quat::from_rotation_x(-FRAC_PI_2);
        assert_vec_close(c * Vec3::X, Vec3::X);
        assert_vec_close(c * Vec3::Y, -Vec3::Z);
        assert_vec_close(c * Vec3::Z, Vec3::Y);
    }

    #[test]
    fn landscape_rotation_90_facing_north_gives_identity_head_quat() {
        // Device rotated 90 deg CCW into landscape, back camera pointing North (+Y_w):
        // +X_dev points Up (+Z_w), +Y_dev points West (-X_w), +Z_dev points South (-Y_w).
        // Let's build q_sensor from those world columns:
        let m = glam::Mat3::from_cols(
            Vec3::Z,  // +X_dev in sensor world
            -Vec3::X, // +Y_dev in sensor world
            -Vec3::Y, // +Z_dev in sensor world
        );
        let q_sensor = Quat::from_mat3(&m);
        let q_head = sensor_to_head_quat(q_sensor, 1);
        assert_vec_close(q_head * Vec3::X, Vec3::X);
        assert_vec_close(q_head * Vec3::Y, Vec3::Y);
        assert_vec_close(q_head * Vec3::Z, Vec3::Z);
    }

    #[test]
    fn predict_ar_rotation_cancels_arbitrary_world_yaw_offset() {
        let ar_world_offset = Quat::from_rotation_y(1.23);
        let gyro_capture = sensor_to_head_quat(Quat::from_rotation_z(0.4), 1);
        let local_yaw_step = Quat::from_rotation_y(-0.15);
        let gyro_latest = (gyro_capture * local_yaw_step).normalize();

        let ar_capture = (ar_world_offset * gyro_capture).normalize();
        let predicted = predict_ar_rotation(ar_capture, gyro_capture, gyro_latest);
        let expected = (ar_world_offset * gyro_latest).normalize();

        assert!(predicted.dot(expected).abs() > 0.9999);
    }

    #[test]
    fn rotation_90_and_270_remain_normalized() {
        let raw = Quat::from_euler(glam::EulerRot::YXZ, 0.3, -0.2, 0.1);
        let q90 = sensor_to_head_quat(raw, 1);
        let q270 = sensor_to_head_quat(raw, 3);
        assert!((q90.length() - 1.0).abs() < 1e-5);
        assert!((q270.length() - 1.0).abs() < 1e-5);
    }
}
