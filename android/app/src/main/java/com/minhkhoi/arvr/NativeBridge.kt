package com.minhkhoi.arvr

object NativeBridge {
    init {
        System.loadLibrary("ar_vr_core")
    }

    external fun nativeInit()
    external fun nativeOnSurfaceCreated(): Int
    external fun nativeOnSurfaceChanged(width: Int, height: Int)
    external fun nativeOnGyroRotation(
        qx: Float,
        qy: Float,
        qz: Float,
        qw: Float,
        displayRotation: Int
    )
    external fun nativeOnArFrame(
        px: Float,
        py: Float,
        pz: Float,
        rx: Float,
        ry: Float,
        rz: Float,
        rw: Float,
        tanHalfH: Float,
        tanHalfV: Float,
        uvs: FloatArray,
        validTracking: Boolean
    )
    external fun nativeOnFallbackCameraFrame(
        tanHalfH: Float,
        tanHalfV: Float,
        uvs: FloatArray
    )
    external fun nativeSetCalibration(fovScale: Float, ipdMeters: Float)
    external fun nativeRecenter()
    external fun nativeDrawFrame()
}
