package com.minhkhoi.arvr

import java.nio.ByteBuffer

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
    external fun nativeSetCalibration(fovScale: Float, ipdMeters: Float)
    external fun nativeSetDrawMode(drawMode: Int)
    external fun nativeClearDrawings()
    external fun nativeRecenter()
    external fun nativeOnHandLandmarks(handCount: Int, packedData: FloatArray): Int
    external fun nativeSetRoomScanMode(mode: Int)
    external fun nativeOnArPlanes(planeCount: Int, packedPlanes: FloatArray)
    external fun nativeOnArPlanePolygons(
        planeCount: Int,
        packedMeta: FloatArray,
        polyCounts: IntArray,
        packedPolyXz: FloatArray
    )
    external fun nativeOnDetectedObjects(
        objectCount: Int,
        packedObjects: FloatArray,
        poseAndTan: FloatArray
    )
    external fun nativeOnPointCloud(pointCount: Int, packedXyzc: FloatArray)
    external fun nativeSpawnPhysicsBody(throwSpeed: Float)
    external fun nativeSetStereoMode(enabled: Boolean)
    external fun nativeOnSurfaceReticle(
        valid: Boolean,
        hitKind: Int,
        px: Float,
        py: Float,
        pz: Float,
        nx: Float,
        ny: Float,
        nz: Float,
        dist: Float,
        outInfo: FloatArray
    )
    external fun nativeDropPhysicsAtReticle()
    external fun nativeThrowPhysicsAtReticle(throwSpeed: Float)
    external fun nativeClearPhysicsBodies()
    external fun nativeGetRoomLabels(outArr: FloatArray): Int
    external fun nativeGetPerfMetrics(outArr: FloatArray)
    external fun nativeGetRenderFrameMs(): Float
    external fun nativeIngestCameraImage(
        yBuf: ByteBuffer,
        uBuf: ByteBuffer,
        vBuf: ByteBuffer,
        yLen: Int,
        uLen: Int,
        vLen: Int,
        width: Int,
        height: Int,
        yRowStride: Int,
        yPixelStride: Int,
        uvRowStride: Int,
        uvPixelStride: Int,
        rotDeg: Int,
        ndcBasis: FloatArray
    ): Long
    external fun nativeConvertLatestFrameRgba(
        outRgbaBuf: ByteBuffer,
        dstW: Int,
        dstH: Int,
        lastSeq: Long,
        updateMotion: Boolean,
        outMeta: FloatArray
    ): Long
    external fun nativeDrawFrame()
}
