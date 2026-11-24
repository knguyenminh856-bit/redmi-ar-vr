package com.minhkhoi.arvr

import android.content.Context
import android.os.Handler
import android.os.SystemClock
import android.util.Log
import com.google.mediapipe.framework.image.ByteBufferImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import java.io.File
import java.nio.ByteBuffer
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max

/**
 * RAM-backed zero-Bitmap hand tracker reading upright RGBA frames from the Rust 2MB YUV LUT ring
 * and throttling GPU palm-detection with a 3-gear adaptive governor so Mali-G57 MC2 never stalls.
 */
class HandTrackerHelper(
    context: Context,
    private val uiHandler: Handler,
    private val onGestureDetected: (Int) -> Unit
) {
    companion object {
        private const val TAG = "HandTracker"
        private const val MODEL_ASSET = "hand_landmarker.task"
        private const val TARGET_W = 320
        private const val TARGET_H = 240
        private const val MAX_HANDS = 2
        private const val LANDMARKS_PER_HAND = 21
        private const val FLOATS_PER_LANDMARK = 5
    }

    val isBusy = AtomicBoolean(false)
    private val isClosed = AtomicBoolean(false)
    private val initStarted = AtomicBoolean(false)
    private val appCtx = context.applicationContext
    private val executor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "HandVisionThread").apply { priority = Thread.NORM_PRIORITY }
    }

    @Volatile
    private var handLandmarker: HandLandmarker? = null

    private var pinnedModelBuf: ByteBuffer? = null

    @Volatile
    var handMode: Int = 0 // 0=TẮT (mặc định tối ưu 3D), 1=TỰ ĐỘNG, 2=NHANH
        private set

    @Volatile
    var aiFps: Float = 0f
        private set

    @Volatile
    var lastInferMs: Int = 0
        private set

    @Volatile
    var delegateLabel: String = "Init"
        private set

    @Volatile
    var lastSkinPermille: Int = 0
        private set

    @Volatile
    var lastMotionPermille: Int = 0
        private set

    @Volatile
    var lastConvertUs: Int = 0
        private set

    private var lastDispatchMs = 0L
    private var lastHandSeenMs = 0L
    private var lastSeq = 0L
    private var lastCompletedNs = 0L
    private var lastTimestampMs = 0L

    private val rgbaDirectBuf: ByteBuffer = ByteBuffer.allocateDirect(TARGET_W * TARGET_H * 4)
    private val metaBuf = FloatArray(19)
    private val packedFloats = FloatArray(MAX_HANDS * LANDMARKS_PER_HAND * FLOATS_PER_LANDMARK)
    private val uprightOpts = ImageProcessingOptions.builder().setRotationDegrees(0).build()

    private fun ensureInitialized() {
        if (!initStarted.compareAndSet(false, true)) return
        executor.execute {
            if (!isClosed.get()) {
                pinnedModelBuf = preloadAssetToDirectBuffer(appCtx, MODEL_ASSET)
                handLandmarker = createLandmarker(appCtx, Delegate.GPU)
                    ?: createLandmarker(appCtx, Delegate.CPU)
            }
        }
    }

    fun setMode(mode: Int) {
        val clamped = mode.coerceIn(0, 2)
        handMode = clamped
        if (clamped == 0) {
            aiFps = 0f
            lastInferMs = 0
            NativeBridge.nativeOnHandLandmarks(0, packedFloats)
        } else {
            ensureInitialized()
        }
    }

    private fun preloadAssetToDirectBuffer(context: Context, assetName: String): ByteBuffer? {
        return try {
            val bytes = context.assets.open(assetName).use { it.readBytes() }
            ByteBuffer.allocateDirect(bytes.size).apply {
                put(bytes)
                rewind()
            }
        } catch (t: Throwable) {
            Log.w(TAG, "RAM model preload failed ($assetName): ${t.message}")
            null
        }
    }

    private fun createLandmarker(context: Context, delegate: Delegate): HandLandmarker? {
        return try {
            val baseBuilder = BaseOptions.builder().setDelegate(delegate)
            val modelBuf = pinnedModelBuf
            if (modelBuf != null) {
                modelBuf.rewind()
                baseBuilder.setModelAssetBuffer(modelBuf)
            } else {
                baseBuilder.setModelAssetPath(MODEL_ASSET)
            }

            if (delegate == Delegate.GPU) {
                val cacheDir = context.codeCacheDir
                if (!cacheDir.exists()) cacheDir.mkdirs()
                val gpuOpts = BaseOptions.DelegateOptions.GpuOptions.builder()
                    .setCachedKernelPath(File(cacheDir, "mp_hand_kernel.bin").absolutePath)
                    .setSerializedModelDir(cacheDir.absolutePath)
                    .setModelToken("hand_lm_v1")
                    .build()
                baseBuilder.setDelegateOptions(gpuOpts)
            }

            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseBuilder.build())
                .setRunningMode(RunningMode.VIDEO)
                .setNumHands(MAX_HANDS)
                .setMinHandDetectionConfidence(0.40f)
                .setMinHandPresenceConfidence(0.40f)
                .setMinTrackingConfidence(0.40f)
                .build()

            val hl = HandLandmarker.createFromOptions(context, options)
            delegateLabel = delegate.name
            Log.i(TAG, "MediaPipe HandLandmarker pinned in RAM ($delegate, ${TARGET_W}x${TARGET_H})")
            hl
        } catch (t: Throwable) {
            Log.w(TAG, "HandLandmarker init failed on $delegate: ${t.message}")
            null
        }
    }

    /**
     * Non-blocking adaptive trigger called after `nativeIngestCameraImage` writes the frame to RAM.
     */
    fun notifyFrameAvailable() {
        val landmarker = handLandmarker ?: return
        if (isClosed.get() || handMode == 0) return

        val nowMs = SystemClock.uptimeMillis()
        val renderMs = NativeBridge.nativeGetRenderFrameMs()
        val minGapMs = when {
            renderMs > 22f -> 110L
            handMode == 2 || (nowMs - lastHandSeenMs) <= 900L -> 42L
            lastSkinPermille >= 18 || lastMotionPermille >= 45 -> 85L
            else -> 240L
        }
        if (nowMs - lastDispatchMs < minGapMs) return
        if (!isBusy.compareAndSet(false, true)) return
        lastDispatchMs = nowMs

        executor.execute {
            try {
                processLatestFrame(landmarker)
            } catch (t: Throwable) {
                Log.w(TAG, "HandLandmarker frame error: ${t.message}")
            } finally {
                isBusy.set(false)
            }
        }
    }

    private fun processLatestFrame(landmarker: HandLandmarker) {
        if (isClosed.get() || handMode == 0) return

        rgbaDirectBuf.position(0)
        val seq = NativeBridge.nativeConvertLatestFrameRgba(
            rgbaDirectBuf,
            TARGET_W,
            TARGET_H,
            lastSeq,
            true,
            metaBuf
        )
        if (seq <= 0L) return
        lastSeq = seq

        lastSkinPermille = metaBuf[0].toInt()
        lastMotionPermille = metaBuf[1].toInt()
        lastConvertUs = metaBuf[2].toInt()
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
            TARGET_W,
            TARGET_H,
            MPImage.IMAGE_FORMAT_RGBA
        ).build()
        val ts = max(lastTimestampMs + 1L, startMs)
        lastTimestampMs = ts

        val result = try {
            landmarker.detectForVideo(mpImage, uprightOpts, ts)
        } finally {
            mpImage.close()
        }

        lastInferMs = (SystemClock.uptimeMillis() - startMs).toInt().coerceAtLeast(1)
        val doneNs = System.nanoTime()
        if (lastCompletedNs != 0L) {
            val dtSec = (doneNs - lastCompletedNs) / 1_000_000_000f
            if (dtSec in 0.005f..2.0f) {
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

        if (validCount > 0) {
            lastHandSeenMs = SystemClock.uptimeMillis()
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
            pinnedModelBuf = null
        }
        executor.shutdown()
    }
}
