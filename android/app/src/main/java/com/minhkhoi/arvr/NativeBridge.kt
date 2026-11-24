package com.minhkhoi.arvr

object NativeBridge {
    init {
        System.loadLibrary("ar_vr_core")
    }

    external fun nativeInit()
    external fun nativeOnSurfaceCreated(): Int
    external fun nativeOnSurfaceChanged(width: Int, height: Int)
    external fun nativeGetGpuInfo(): String
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
    external fun nativeOnDepthFrame(
        width: Int,
        height: Int,
        packedBytes: ByteArray,
        d0x: Float,
        d0y: Float,
        dxx: Float,
        dxy: Float,
        dyx: Float,
        dyy: Float
    )
    external fun nativeSetCalibration(fovScale: Float, ipdMeters: Float)
    external fun nativeSetFeatures(depthOcclusion: Boolean, drawMode: Int)
    external fun nativeClearDrawings()
    external fun nativeRecenter()
    external fun nativeOnHandLandmarks(handCount: Int, packedData: FloatArray): Int
    external fun nativeGetPerfMetrics(outArr: FloatArray)
    external fun nativeDrawFrame()
}
