package com.minhkhoi.arvr

import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.google.mediapipe.framework.image.ByteBufferImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import java.nio.ByteBuffer
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs

/**
 * RAM-backed CPU room-object detector reading pre-rotated 256x192 RGBA frames directly from
 * the Rust 2MB YUV LUT ring at ~1.5 Hz and pinning detections in 3D spatial RAM cache.
 */
class RoomScannerHelper(private val context: Context) {

    companion object {
        private const val TAG = "RoomScannerHelper"
        private const val MODEL_ASSET = "efficientdet_lite0.tflite"
        private const val MAX_OBJECTS = 6
        private const val FLOATS_PER_OBJECT = 6
        private const val MIN_INTERVAL_MS = 650L
        private const val DETECT_W = 256
        private const val DETECT_H = 192
    }

    private val executor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "RoomObjectThread").apply { priority = Thread.NORM_PRIORITY - 1 }
    }

    val isBusy = AtomicBoolean(false)
    private val isClosed = AtomicBoolean(false)

    @Volatile
    private var objectDetector: ObjectDetector? = null

    private var pinnedModelBuf: ByteBuffer? = null

    @Volatile
    var lastInferMs: Int = 0
        private set

    @Volatile
    var lastObjectCount: Int = 0
        private set

    private var lastSubmittedMs: Long = 0L
    private var lastSeq: Long = 0L

    private val rgbaDirectBuf: ByteBuffer = ByteBuffer.allocateDirect(DETECT_W * DETECT_H * 4)
    private val metaBuf = FloatArray(19)
    private val poseAndTanBuf = FloatArray(9)
    private val packedFloats = FloatArray(MAX_OBJECTS * FLOATS_PER_OBJECT)
    private val uprightOpts = ImageProcessingOptions.builder().setRotationDegrees(0).build()

    fun initAsync() {
        val appCtx = context.applicationContext
        executor.execute {
            if (isClosed.get()) return@execute
            try {
                val modelBytes = appCtx.assets.open(MODEL_ASSET).use { it.readBytes() }
                val directModel = ByteBuffer.allocateDirect(modelBytes.size).apply {
                    put(modelBytes)
                    rewind()
                }
                pinnedModelBuf = directModel

                val baseOptions = BaseOptions.builder()
                    .setDelegate(Delegate.CPU)
                    .setModelAssetBuffer(directModel)
                    .build()

                val options = ObjectDetector.ObjectDetectorOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setRunningMode(RunningMode.IMAGE)
                    .setMaxResults(MAX_OBJECTS)
                    .setScoreThreshold(0.33f)
                    .build()

                objectDetector = ObjectDetector.createFromOptions(appCtx, options)
                Log.i(TAG, "ObjectDetector pinned in RAM (CPU INT8, ${DETECT_W}x${DETECT_H})")
            } catch (t: Throwable) {
                Log.w(TAG, "ObjectDetector init failed: ${t.message}", t)
            }
        }
    }

    fun resetMetrics() {
        lastInferMs = 0
        lastObjectCount = 0
    }

    fun notifyFrameAvailable() {
        val detector = objectDetector ?: return
        if (isClosed.get()) return

        val nowMs = SystemClock.uptimeMillis()
        if (nowMs - lastSubmittedMs < MIN_INTERVAL_MS) return
        if (!isBusy.compareAndSet(false, true)) return
        lastSubmittedMs = nowMs

        executor.execute {
            try {
                processLatestFrame(detector)
            } catch (t: Throwable) {
                Log.w(TAG, "ObjectDetector frame error: ${t.message}")
            } finally {
                isBusy.set(false)
            }
        }
    }

    private fun processLatestFrame(detector: ObjectDetector) {
        if (isClosed.get()) return

        rgbaDirectBuf.position(0)
        val seq = NativeBridge.nativeConvertLatestFrameRgba(
            rgbaDirectBuf,
            DETECT_W,
            DETECT_H,
            lastSeq,
            false,
            metaBuf
        )
        if (seq <= 0L) return
        lastSeq = seq

        // Extract capture pose & camera tangents for 3D raycast spatial pinning
        for (i in 0 until 9) {
            poseAndTanBuf[i] = metaBuf[3 + i]
        }
        val p0x = metaBuf[12]
        val p0y = metaBuf[13]
        val pxx = metaBuf[14]
        val pxy = metaBuf[15]
        val pyx = metaBuf[16]
        val pyy = metaBuf[17]
        val rotDeg = metaBuf[18].toInt()

        rgbaDirectBuf.position(0)
        val startMs = SystemClock.uptimeMillis()
        val mpImage = ByteBufferImageBuilder(
            rgbaDirectBuf,
            DETECT_W,
            DETECT_H,
            MPImage.IMAGE_FORMAT_RGBA
        ).build()

        val result = try {
            detector.detect(mpImage, uprightOpts)
        } finally {
            mpImage.close()
        }
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
            val cxRot = (box.centerX() / DETECT_W.toFloat()).coerceIn(0.02f, 0.98f)
            val cyRot = (box.centerY() / DETECT_H.toFloat()).coerceIn(0.02f, 0.98f)
            val bwNorm = (box.width() / DETECT_W.toFloat()).coerceIn(0.04f, 0.95f)
            val bhNorm = (box.height() / DETECT_H.toFloat()).coerceIn(0.04f, 0.95f)

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
        NativeBridge.nativeOnDetectedObjects(validCount, packedFloats, poseAndTanBuf)
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
            else -> if (score >= 0.45f) 19 else null
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
            pinnedModelBuf = null
        }
        executor.shutdown()
    }
}
