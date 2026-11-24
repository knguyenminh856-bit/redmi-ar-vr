package com.minhkhoi.arvr

import android.content.Context
import android.graphics.Bitmap
import android.media.Image
import android.os.SystemClock
import android.util.Log
import android.view.Surface
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs

/**
 * Lightweight background room-object detector running EfficientDet-Lite0 INT8 on CPU at ~2.6 Hz
 * so 120Hz passthrough and GPU hand tracking remain unaffected.
 */
class RoomScannerHelper(private val context: Context) {

    companion object {
        private const val TAG = "RoomScannerHelper"
        private const val MAX_OBJECTS = 6
        private const val FLOATS_PER_OBJECT = 6
        private const val MIN_INTERVAL_MS = 380L
        private const val DETECT_W = 320
    }

    private val executor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "RoomObjectThread").apply { priority = Thread.NORM_PRIORITY - 1 }
    }

    val isBusy = AtomicBoolean(false)
    private val isClosed = AtomicBoolean(false)

    @Volatile
    private var objectDetector: ObjectDetector? = null

    @Volatile
    var lastInferMs: Int = 0
        private set

    @Volatile
    var lastObjectCount: Int = 0
        private set

    @Volatile
    private var lastSubmittedMs: Long = 0L

    private var yCopy = ByteArray(640 * 480)
    private var uCopy = ByteArray(640 * 480)
    private var vCopy = ByteArray(640 * 480)
    private var argbPixels = IntArray(DETECT_W * 240)
    private var reusableBitmap: Bitmap? = null
    private val packedFloats = FloatArray(MAX_OBJECTS * FLOATS_PER_OBJECT)

    fun initAsync() {
        executor.execute {
            if (isClosed.get()) return@execute
            try {
                val baseOptions = BaseOptions.builder()
                    .setDelegate(Delegate.CPU)
                    .setModelAssetPath("efficientdet_lite0.tflite")
                    .build()

                val options = ObjectDetector.ObjectDetectorOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setRunningMode(RunningMode.IMAGE)
                    .setMaxResults(MAX_OBJECTS)
                    .setScoreThreshold(0.40f)
                    .build()

                objectDetector = ObjectDetector.createFromOptions(context, options)
                Log.i(TAG, "MediaPipe ObjectDetector initialized on CPU (INT8)")
            } catch (t: Throwable) {
                Log.w(TAG, "ObjectDetector init failed: ${t.message}", t)
            }
        }
    }

    /**
     * Copies YUV planes quickly after rewinding positions (safe to share an Image with HandTrackerHelper),
     * throttled to <= 2.6 Hz.
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
        if (isClosed.get() || objectDetector == null) return
        val nowMs = SystemClock.uptimeMillis()
        if (nowMs - lastSubmittedMs < MIN_INTERVAL_MS) return

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
            lastSubmittedMs = nowMs

            executor.execute {
                try {
                    processCopiedYuv(
                        srcW, srcH, yLen, uLen, vLen,
                        yRowStride, yPixelStride, uvRowStride, uvPixelStride,
                        displayRotation, p0x, p0y, pxx, pxy, pyx, pyy
                    )
                } catch (t: Throwable) {
                    Log.w(TAG, "ObjectDetector frame error: ${t.message}")
                } finally {
                    isBusy.set(false)
                }
            }
        } catch (t: Throwable) {
            isBusy.set(false)
            Log.w(TAG, "YUV copy failed: ${t.message}")
        }
    }

    private fun processCopiedYuv(
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
        val detector = objectDetector ?: return
        if (isClosed.get()) return
        val startMs = SystemClock.uptimeMillis()

        val dstW = DETECT_W
        val dstH = (dstW * srcH / srcW).coerceIn(120, 320)
        if (argbPixels.size != dstW * dstH) {
            argbPixels = IntArray(dstW * dstH)
        }
        val bmp = if (reusableBitmap != null && reusableBitmap!!.width == dstW && reusableBitmap!!.height == dstH) {
            reusableBitmap!!
        } else {
            reusableBitmap?.recycle()
            Bitmap.createBitmap(dstW, dstH, Bitmap.Config.ARGB_8888).also { reusableBitmap = it }
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

        val result = detector.detect(mpImage, imgOpts)
        lastInferMs = (SystemClock.uptimeMillis() - startMs).toInt().coerceAtLeast(1)

        val dxX = pxx - p0x
        val dxY = pxy - p0y
        val dyX = pyx - p0x
        val dyY = pyy - p0y

        var validCount = 0
        for (detection in result.detections()) {
            if (validCount >= MAX_OBJECTS) break
            val cat = detection.categories().firstOrNull() ?: continue
            val rawName = cat.categoryName()?.lowercase()?.trim() ?: continue
            val score = cat.score()
            val kindId = mapCocoCategoryToKindId(rawName, score) ?: continue

            val box = detection.boundingBox()
            val cxRot = (box.centerX() / dstW.toFloat()).coerceIn(0.02f, 0.98f)
            val cyRot = (box.centerY() / dstH.toFloat()).coerceIn(0.02f, 0.98f)
            val bwNorm = (box.width() / dstW.toFloat()).coerceIn(0.04f, 0.95f)
            val bhNorm = (box.height() / dstH.toFloat()).coerceIn(0.04f, 0.95f)

            val uImg = if (rotDeg == 180) 1f - cxRot else cxRot
            val vImg = if (rotDeg == 180) 1f - cyRot else cyRot

            val ndcX = p0x + uImg * dxX + vImg * dyX
            val ndcY = p0y + uImg * dxY + vImg * dyY
            val ndcW = abs(bwNorm * dxX) + abs(bhNorm * dyX)
            val ndcH = abs(bwNorm * dxY) + abs(bhNorm * dyY)

            val base = validCount * FLOATS_PER_OBJECT
            packedFloats[base] = kindId.toFloat()
            packedFloats[base + 1] = ndcX
            packedFloats[base + 2] = ndcY
            packedFloats[base + 3] = ndcW
            packedFloats[base + 4] = ndcH
            packedFloats[base + 5] = score
            validCount++
        }

        lastObjectCount = validCount
        NativeBridge.nativeOnDetectedObjects(validCount, packedFloats)
    }

    private fun mapCocoCategoryToKindId(label: String, score: Float): Int? {
        if (label == "person") return null
        return when (label) {
            "dining table", "desk", "table" -> 11
            "chair" -> 12
            "couch", "sofa" -> 13
            "bed" -> 14
            "tv", "monitor", "laptop" -> 15
            "door", "cabinet", "refrigerator", "oven", "microwave", "bookcase" -> 16
            "bottle", "cup", "mug", "wine glass", "bowl" -> 17
            "potted plant", "vase" -> 18
            "keyboard", "mouse", "remote", "cell phone", "book", "clock",
            "backpack", "handbag", "suitcase", "bench", "toilet", "sink", "toaster" -> 19
            else -> if (score >= 0.50f) 19 else null
        }
    }

    fun close() {
        isClosed.set(true)
        executor.execute {
            try {
                objectDetector?.close()
            } catch (_: Throwable) {
            }
            objectDetector = null
            reusableBitmap?.recycle()
            reusableBitmap = null
        }
        executor.shutdown()
    }
}
