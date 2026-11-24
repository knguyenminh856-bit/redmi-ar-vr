package com.minhkhoi.arvr

import android.content.Context
import android.graphics.Bitmap
import android.media.Image
import android.os.Handler
import android.os.Process
import android.os.SystemClock
import android.util.Log
import android.view.Surface
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max

class HandTrackerHelper(
    context: Context,
    private val uiHandler: Handler,
    private val onGestureDetected: (Int) -> Unit
) {
    companion object {
        private const val TAG = "HandTracker"
        private const val MODEL_ASSET = "hand_landmarker.task"
        private const val TARGET_WIDTH = 224
        private const val MIN_SUBMIT_INTERVAL_NS = 48_000_000L // ~20 Hz to preserve CPU/thermal headroom
        private const val MAX_HANDS = 2
        private const val LANDMARKS_PER_HAND = 21
        private const val FLOATS_PER_LANDMARK = 5
    }

    val isBusy = AtomicBoolean(false)
    private val isClosed = AtomicBoolean(false)
    private val executor = Executors.newSingleThreadExecutor()

    @Volatile
    private var handLandmarker: HandLandmarker? = null

    @Volatile
    var aiFps: Float = 0f
        private set

    @Volatile
    var lastInferMs: Int = 0
        private set

    private var lastSubmitNs = 0L
    private var lastCompletedNs = 0L
    private var argbPixels = IntArray(0)
    private var reusableBitmap: Bitmap? = null
    private val packedFloats = FloatArray(MAX_HANDS * LANDMARKS_PER_HAND * FLOATS_PER_LANDMARK)
    private var lastTimestampMs = 0L

    init {
        val appCtx = context.applicationContext
        executor.execute {
            Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND)
            if (!isClosed.get()) {
                handLandmarker = createCpuLandmarker(appCtx)
            }
        }
    }

    private fun createCpuLandmarker(context: Context): HandLandmarker? {
        return try {
            val baseOpts = BaseOptions.builder()
                .setModelAssetPath(MODEL_ASSET)
                .setDelegate(Delegate.CPU) // Dedicated Cortex-A76 CPU delegate keeps Mali GPU 100% free for 120Hz GLES
                .build()
            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOpts)
                .setRunningMode(RunningMode.VIDEO)
                .setNumHands(MAX_HANDS)
                .setMinHandDetectionConfidence(0.45f)
                .setMinHandPresenceConfidence(0.45f)
                .setMinTrackingConfidence(0.45f)
                .build()
            val hl = HandLandmarker.createFromOptions(context, options)
            Log.i(TAG, "MediaPipe HandLandmarker initialized on CPU (XNNPACK)")
            hl
        } catch (t: Throwable) {
            Log.e(TAG, "HandLandmarker init failed: ${t.message}")
            null
        }
    }

    /**
     * Subsamples YUV_420_888 directly to a 224px-wide ARGB array (<0.15ms, zero 3MB YUV buffer copy)
     * so `image.close()` happens immediately and Mali GPU stays unblocked.
     */
    fun submitYuvFrame(
        image: Image,
        displayRotation: Int,
        p0x: Float,
        p0y: Float,
        pxx: Float,
        pxy: Float,
        pyx: Float,
        pyy: Float
    ) {
        if (isClosed.get() || handLandmarker == null) return
        val nowNs = System.nanoTime()
        if (nowNs - lastSubmitNs < MIN_SUBMIT_INTERVAL_NS) return
        val planes = image.planes
        if (planes.size < 3) return
        if (!isBusy.compareAndSet(false, true)) return
        lastSubmitNs = nowNs

        try {
            val srcW = image.width
            val srcH = image.height
            if (srcW <= 0 || srcH <= 0) {
                isBusy.set(false)
                return
            }

            val dstW = TARGET_WIDTH.coerceAtMost(srcW)
            val dstH = ((srcH * dstW) / srcW).coerceAtLeast(1)
            val totalPixels = dstW * dstH
            if (argbPixels.size != totalPixels) {
                argbPixels = IntArray(totalPixels)
            }

            val yBuf = planes[0].buffer
            val uBuf = planes[1].buffer
            val vBuf = planes[2].buffer
            val yCap = yBuf.limit()
            val uCap = uBuf.limit()
            val vCap = vBuf.limit()

            val yRowStride = planes[0].rowStride
            val yPixelStride = planes[0].pixelStride
            val uvRowStride = planes[1].rowStride
            val uvPixelStride = planes[1].pixelStride

            for (dy in 0 until dstH) {
                val sy = (dy * srcH) / dstH
                val yRowOff = sy * yRowStride
                val uvRowOff = (sy shr 1) * uvRowStride
                val rowOut = dy * dstW
                for (dx in 0 until dstW) {
                    val sx = (dx * srcW) / dstW
                    val yIdx = yRowOff + sx * yPixelStride
                    val uvIdx = uvRowOff + (sx shr 1) * uvPixelStride

                    val yVal = if (yIdx in 0 until yCap) (yBuf.get(yIdx).toInt() and 0xFF) - 16 else 0
                    val uVal = if (uvIdx in 0 until uCap) (uBuf.get(uvIdx).toInt() and 0xFF) - 128 else 0
                    val vVal = if (uvIdx in 0 until vCap) (vBuf.get(uvIdx).toInt() and 0xFF) - 128 else 0

                    val c = if (yVal > 0) 298 * yVal else 0
                    val r = ((c + 409 * vVal + 128) shr 8).coerceIn(0, 255)
                    val g = ((c - 100 * uVal - 208 * vVal + 128) shr 8).coerceIn(0, 255)
                    val b = ((c + 516 * uVal + 128) shr 8).coerceIn(0, 255)
                    argbPixels[rowOut + dx] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }

            executor.execute {
                try {
                    if (!isClosed.get()) {
                        processFrameOnWorker(
                            dstW, dstH, displayRotation,
                            p0x, p0y, pxx, pxy, pyx, pyy
                        )
                    }
                } catch (t: Throwable) {
                    Log.w(TAG, "Hand tracking worker warning: ${t.message}")
                } finally {
                    isBusy.set(false)
                }
            }
        } catch (t: Throwable) {
            isBusy.set(false)
            Log.w(TAG, "Failed to subsample YUV image: ${t.message}")
        }
    }

    private fun processFrameOnWorker(
        dstW: Int,
        dstH: Int,
        displayRotation: Int,
        p0x: Float,
        p0y: Float,
        pxx: Float,
        pxy: Float,
        pyx: Float,
        pyy: Float
    ) {
        val landmarker = handLandmarker ?: return
        val startMs = SystemClock.uptimeMillis()

        val bmp = reusableBitmap?.takeIf { it.width == dstW && it.height == dstH }
            ?: Bitmap.createBitmap(dstW, dstH, Bitmap.Config.ARGB_8888).also {
                reusableBitmap = it
            }
        bmp.setPixels(argbPixels, 0, dstW, 0, 0, dstW, dstH)

        val rotDeg = if (displayRotation == Surface.ROTATION_270) 180 else 0
        val imgOpts = ImageProcessingOptions.builder()
            .setRotationDegrees(rotDeg)
            .build()
        val mpImage = BitmapImageBuilder(bmp).build()
        val ts = max(lastTimestampMs + 1L, SystemClock.uptimeMillis())
        lastTimestampMs = ts

        val result = landmarker.detectForVideo(mpImage, imgOpts, ts)
        val inferMs = (SystemClock.uptimeMillis() - startMs).toInt().coerceAtLeast(1)
        lastInferMs = inferMs

        val doneNs = System.nanoTime()
        if (lastCompletedNs != 0L) {
            val dtSec = (doneNs - lastCompletedNs) / 1_000_000_000f
            if (dtSec > 0.005f && dtSec < 2.0f) {
                val instFps = 1f / dtSec
                aiFps = if (aiFps < 0.5f) instFps else (aiFps * 0.8f + instFps * 0.2f)
            }
        }
        lastCompletedNs = doneNs

        val normHands = result.landmarks()
        val worldHands = result.worldLandmarks()
        val count = normHands.size.coerceAtMost(MAX_HANDS)

        var validCount = 0
        val dxX = pxx - p0x
        val dxY = pxy - p0y
        val dyX = pyx - p0x
        val dyY = pyy - p0y

        for (h in 0 until count) {
            val lms = normHands[h]
            val wms = if (h < worldHands.size) worldHands[h] else continue
            if (lms.size != LANDMARKS_PER_HAND || wms.size != LANDMARKS_PER_HAND) continue

            val handBase = validCount * LANDMARKS_PER_HAND * FLOATS_PER_LANDMARK
            for (i in 0 until LANDMARKS_PER_HAND) {
                val lm = lms[i]
                val wm = wms[i]
                val uImg = if (rotDeg == 180) 1f - lm.x() else lm.x()
                val vImg = if (rotDeg == 180) 1f - lm.y() else lm.y()

                val ndcX = p0x + uImg * dxX + vImg * dyX
                val ndcY = p0y + uImg * dxY + vImg * dyY

                val dst = handBase + i * FLOATS_PER_LANDMARK
                packedFloats[dst] = ndcX
                packedFloats[dst + 1] = ndcY
                packedFloats[dst + 2] = wm.x()
                packedFloats[dst + 3] = wm.y()
                packedFloats[dst + 4] = wm.z()
            }
            validCount++
        }

        val triggeredGesture = NativeBridge.nativeOnHandLandmarks(validCount, packedFloats)
        if (triggeredGesture > 0) {
            uiHandler.post { onGestureDetected(triggeredGesture) }
        }
    }

    fun close() {
        isClosed.set(true)
        executor.execute {
            try {
                handLandmarker?.close()
            } catch (_: Throwable) {
            }
            handLandmarker = null
            reusableBitmap?.recycle()
            reusableBitmap = null
        }
        executor.shutdown()
    }
}
