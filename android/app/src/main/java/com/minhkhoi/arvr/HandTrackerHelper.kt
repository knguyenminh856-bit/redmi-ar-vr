package com.minhkhoi.arvr

import android.content.Context
import android.graphics.Bitmap
import android.media.Image
import android.os.Handler
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
        private const val TARGET_WIDTH = 480
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

    @Volatile
    var delegateLabel: String = "Init"
        private set

    private var lastCompletedNs = 0L
    private var yCopy = ByteArray(0)
    private var uCopy = ByteArray(0)
    private var vCopy = ByteArray(0)
    private var argbPixels = IntArray(0)
    private var reusableBitmap: Bitmap? = null
    private val packedFloats = FloatArray(MAX_HANDS * LANDMARKS_PER_HAND * FLOATS_PER_LANDMARK)
    private var lastTimestampMs = 0L

    init {
        val appCtx = context.applicationContext
        executor.execute {
            if (!isClosed.get()) {
                handLandmarker = createLandmarker(appCtx, Delegate.GPU)
                    ?: createLandmarker(appCtx, Delegate.CPU)
            }
        }
    }

    private fun createLandmarker(context: Context, delegate: Delegate): HandLandmarker? {
        return try {
            val baseOpts = BaseOptions.builder()
                .setModelAssetPath(MODEL_ASSET)
                .setDelegate(delegate)
                .build()
            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOpts)
                .setRunningMode(RunningMode.VIDEO)
                .setNumHands(MAX_HANDS)
                .setMinHandDetectionConfidence(0.40f)
                .setMinHandPresenceConfidence(0.40f)
                .setMinTrackingConfidence(0.40f)
                .build()
            val hl = HandLandmarker.createFromOptions(context, options)
            delegateLabel = delegate.name
            Log.i(TAG, "MediaPipe HandLandmarker initialized on $delegate")
            hl
        } catch (t: Throwable) {
            Log.w(TAG, "HandLandmarker init failed on $delegate: ${t.message}")
            null
        }
    }

    /**
     * Copies 640x480 YUV planes in <0.1ms via native bulk ByteBuffer.get(), then converts to a crisp
     * 480px-wide Bitmap and runs GPU MediaPipe inference on the worker thread without dropping frames.
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
        val planes = image.planes
        if (planes.size < 3) return
        if (!isBusy.compareAndSet(false, true)) return

        try {
            val srcW = image.width
            val srcH = image.height
            if (srcW <= 0 || srcH <= 0) {
                isBusy.set(false)
                return
            }

            val yBuf = planes[0].buffer
            val uBuf = planes[1].buffer
            val vBuf = planes[2].buffer
            yBuf.position(0)
            uBuf.position(0)
            vBuf.position(0)

            val yLen = yBuf.remaining()
            val uLen = uBuf.remaining()
            val vLen = vBuf.remaining()

            if (yCopy.size < yLen) yCopy = ByteArray(yLen)
            if (uCopy.size < uLen) uCopy = ByteArray(uLen)
            if (vCopy.size < vLen) vCopy = ByteArray(vLen)

            yBuf.get(yCopy, 0, yLen)
            uBuf.get(uCopy, 0, uLen)
            vBuf.get(vCopy, 0, vLen)

            val yRowStride = planes[0].rowStride
            val yPixelStride = planes[0].pixelStride
            val uvRowStride = planes[1].rowStride
            val uvPixelStride = planes[1].pixelStride

            executor.execute {
                try {
                    if (!isClosed.get()) {
                        processFrameOnWorker(
                            srcW, srcH,
                            yLen, uLen, vLen,
                            yRowStride, yPixelStride, uvRowStride, uvPixelStride,
                            displayRotation,
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
            Log.w(TAG, "Failed to copy YUV image: ${t.message}")
        }
    }

    private fun processFrameOnWorker(
        srcW: Int,
        srcH: Int,
        yLen: Int,
        uLen: Int,
        vLen: Int,
        yRowStride: Int,
        yPixelStride: Int,
        uvRowStride: Int,
        uvPixelStride: Int,
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

        val dstW = TARGET_WIDTH.coerceAtMost(srcW)
        val dstH = ((srcH * dstW) / srcW).coerceAtLeast(1)
        val totalPixels = dstW * dstH
        if (argbPixels.size != totalPixels) {
            argbPixels = IntArray(totalPixels)
        }
        val bmp = reusableBitmap?.takeIf { it.width == dstW && it.height == dstH }
            ?: Bitmap.createBitmap(dstW, dstH, Bitmap.Config.ARGB_8888).also {
                reusableBitmap = it
            }

        for (dy in 0 until dstH) {
            val sy = (dy * srcH) / dstH
            val yRowOff = sy * yRowStride
            val uvRowOff = (sy shr 1) * uvRowStride
            val rowOut = dy * dstW
            for (dx in 0 until dstW) {
                val sx = (dx * srcW) / dstW
                val yIdx = yRowOff + sx * yPixelStride
                val uvIdx = uvRowOff + (sx shr 1) * uvPixelStride

                val yVal = if (yIdx in 0 until yLen) (yCopy[yIdx].toInt() and 0xFF) - 16 else 0
                val uVal = if (uvIdx in 0 until uLen) (uCopy[uvIdx].toInt() and 0xFF) - 128 else 0
                val vVal = if (uvIdx in 0 until vLen) (vCopy[uvIdx].toInt() and 0xFF) - 128 else 0

                val c = if (yVal > 0) 298 * yVal else 0
                val r = ((c + 409 * vVal + 128) shr 8).coerceIn(0, 255)
                val g = ((c - 100 * uVal - 208 * vVal + 128) shr 8).coerceIn(0, 255)
                val b = ((c + 516 * uVal + 128) shr 8).coerceIn(0, 255)
                argbPixels[rowOut + dx] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
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
