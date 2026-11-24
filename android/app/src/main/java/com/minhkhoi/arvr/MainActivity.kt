package com.minhkhoi.arvr

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.SurfaceTexture
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.media.ImageReader
import android.opengl.GLSurfaceView
import android.os.BatteryManager
import android.os.Build
import android.os.Bundle
import android.os.Debug
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.util.Log
import android.util.Size
import android.util.TypedValue
import android.view.Gravity
import android.view.KeyEvent
import android.view.Surface
import android.view.View
import android.view.WindowInsets
import android.view.WindowInsetsController
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import com.google.ar.core.CameraConfig
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Config
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.EnumSet
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import javax.microedition.khronos.egl.EGL10
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.egl.EGLDisplay
import javax.microedition.khronos.opengles.GL10
import kotlin.math.abs

class MainActivity : Activity(), SensorEventListener {

    companion object {
        private const val TAG = "RedmiARVR"
        private const val REQ_CAMERA = 1001
        private const val PREFS_NAME = "vr_calibration"
        private const val KEY_FOV_PCT = "fov_pct"
        private const val KEY_IPD_MM = "ipd_mm"
    }

    private lateinit var glView: GLSurfaceView
    private lateinit var sensorManager: SensorManager
    private var rotSensor: Sensor? = null
    private val quatBuf = FloatArray(4)

    private val mainHandler = Handler(Looper.getMainLooper())
    private val sessionLock = Any()

    // ARCore session (primary 6DoF tracking)
    private var arSession: Session? = null
    private var arNeedsTextureBind = false
    private var arInitAttempted = false
    @Volatile
    private var isUsingArCore = false

    // Camera2 fallback (when Google Play Services for AR is unavailable)
    private var cameraThread: HandlerThread? = null
    private var cameraHandler: Handler? = null
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var fallbackSurfaceTexture: SurfaceTexture? = null
    private var fallbackPreviewSurface: Surface? = null
    private var fallbackImageReader: ImageReader? = null
    private val fallbackFrameAvailable = AtomicBoolean(false)
    private var fallbackCameraActive = false
    private var fallbackTanHalfH = 0.67f
    private var fallbackTanHalfV = 0.50f
    private val fallbackTexMtx = FloatArray(16)

    private lateinit var handTracker: HandTrackerHelper
    private lateinit var appUpdater: AppUpdater
    private val imgBasis = floatArrayOf(0f, 0f, 1f, 0f, 0f, 1f)
    private val ndcBasis = FloatArray(6)

    // GL state synced with render thread
    @Volatile
    private var cameraTexId: Int = 0
    private var viewportDirty = false
    private var viewWidth = 2400
    private var viewHeight = 1080
    @Volatile
    private var displayRotation = Surface.ROTATION_90
    private var displayModeW = 2400
    private var displayModeH = 1080
    private var displayHz = 120

    // Pre-allocated buffers for zero-GC 120Hz rendering
    private val ndcQuadBuf: FloatBuffer = ByteBuffer.allocateDirect(8 * 4)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()
        .apply {
            put(floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f))
            position(0)
        }
    private val uvOutBuf: FloatBuffer = ByteBuffer.allocateDirect(8 * 4)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()
    private val uvArray = FloatArray(8)
    private val arProjMat = FloatArray(16)
    private var lastArUpdateNs = 0L
    private var lastArTimestampNs = 0L

    // Feature toggles, OTA Updater & UI
    private var fovPct = 78
    private var ipdMm = 64
    @Volatile
    private var drawMode = 0 // 0=Off, 1=1 Finger, 2=5 Fingers
    private var telemetryVisible = false
    private var availableUpdate: UpdateCheckResult? = null
    private var pendingInstallPermissionResume = false

    private lateinit var panelLayout: LinearLayout
    private var recenterRunnable: Runnable? = null
    private lateinit var leftGesturePopup: TextView
    private lateinit var rightGesturePopup: TextView
    private var hidePopupRunnable: Runnable? = null
    private lateinit var leftTelemetryCard: TextView
    private lateinit var rightTelemetryCard: TextView
    private lateinit var drawToggleBtn: Button
    private lateinit var checkUpdateBtn: Button
    private lateinit var installUpdateBtn: Button
    private lateinit var telemetryToggleBtn: Button
    private val metricsBuf = FloatArray(8)
    private var cachedGpuName = ""
    private var telemetryRunnable: Runnable? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        NativeBridge.nativeInit()
        handTracker = HandTrackerHelper(this, mainHandler) { code ->
            showStereoGesturePopup(code)
        }
        appUpdater = AppUpdater(mainHandler)

        loadCalibration()
        NativeBridge.nativeSetDrawMode(drawMode)
        configureHighRefreshDisplayAndKeepOn()

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        rotSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
            ?: sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

        val root = FrameLayout(this).apply {
            setBackgroundColor(Color.BLACK)
        }

        glView = GLSurfaceView(this).apply {
            setEGLContextClientVersion(3)
            setEGLConfigChooser(FastEglConfigChooser())
            preserveEGLContextOnPause = true
            setRenderer(ArVrGlRenderer())
            renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        }
        root.addView(
            glView,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        )

        addStereoHudAndPopupOverlay(root)
        addTopControlBar(root)
        addCalibrationOverlay(root)
        setContentView(root)
        applyImmersiveFullscreen()

        if (!hasCameraPermission()) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), REQ_CAMERA)
        }

        mainHandler.postDelayed({ performUpdateCheck(manual = false) }, 2500L)
    }

    private fun loadCalibration() {
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        fovPct = prefs.getInt(KEY_FOV_PCT, 78).coerceIn(50, 100)
        ipdMm = prefs.getInt(KEY_IPD_MM, 64).coerceIn(56, 72)
        NativeBridge.nativeSetCalibration(fovPct / 100f, ipdMm / 1000f)
    }

    private fun saveCalibration() {
        getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putInt(KEY_FOV_PCT, fovPct)
            .putInt(KEY_IPD_MM, ipdMm)
            .apply()
        NativeBridge.nativeSetCalibration(fovPct / 100f, ipdMm / 1000f)
    }

    private fun configureHighRefreshDisplayAndKeepOn() {
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        val currentDisplay = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            display
        } else {
            @Suppress("DEPRECATION")
            windowManager.defaultDisplay
        } ?: return

        val curMode = currentDisplay.mode
        val bestMode = currentDisplay.supportedModes
            .filter {
                it.physicalWidth == curMode.physicalWidth &&
                    it.physicalHeight == curMode.physicalHeight
            }
            .maxByOrNull { it.refreshRate } ?: curMode

        displayModeW = bestMode.physicalWidth
        displayModeH = bestMode.physicalHeight
        displayHz = bestMode.refreshRate.toInt()

        if (bestMode.modeId != curMode.modeId) {
            val attrs = window.attributes
            attrs.preferredDisplayModeId = bestMode.modeId
            window.attributes = attrs
        }
        displayRotation = currentDisplay.rotation
    }

    private fun applyImmersiveFullscreen() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.setDecorFitsSystemWindows(false)
            window.insetsController?.let { controller ->
                controller.hide(WindowInsets.Type.statusBars() or WindowInsets.Type.navigationBars())
                controller.systemBarsBehavior =
                    WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            }
        } else {
            @Suppress("DEPRECATION")
            window.decorView.systemUiVisibility = (
                View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                    or View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                    or View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                    or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                    or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                    or View.SYSTEM_UI_FLAG_FULLSCREEN
                )
        }
    }

    private fun addStereoHudAndPopupOverlay(root: FrameLayout) {
        val stereoRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            weightSum = 2f
            isClickable = false
            isFocusable = false
        }

        leftGesturePopup = createPopupPill(nasalShiftPx = 8)
        rightGesturePopup = createPopupPill(nasalShiftPx = -8)
        leftTelemetryCard = createTelemetryCard(nasalShiftPx = 6)
        rightTelemetryCard = createTelemetryCard(nasalShiftPx = -6)

        val leftPane = FrameLayout(this).apply {
            isClickable = false
            isFocusable = false
            addView(leftTelemetryCard)
            addView(leftGesturePopup)
        }
        val rightPane = FrameLayout(this).apply {
            isClickable = false
            isFocusable = false
            addView(rightTelemetryCard)
            addView(rightGesturePopup)
        }

        val halfParams = LinearLayout.LayoutParams(
            0,
            LinearLayout.LayoutParams.MATCH_PARENT,
            1f
        )
        stereoRow.addView(leftPane, halfParams)
        stereoRow.addView(rightPane, halfParams)

        root.addView(
            stereoRow,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        )
    }

    private fun createPopupPill(nasalShiftPx: Int): TextView {
        val pillBg = GradientDrawable().apply {
            cornerRadius = 36f
            setColor(Color.argb(226, 14, 20, 34))
            setStroke(3, Color.parseColor("#00E5FF"))
        }
        return TextView(this).apply {
            visibility = View.GONE
            background = pillBg
            setTextColor(Color.WHITE)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 14.5f)
            typeface = Typeface.DEFAULT_BOLD
            setPadding(28, 16, 28, 16)
            gravity = Gravity.CENTER
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.WRAP_CONTENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.TOP or Gravity.CENTER_HORIZONTAL
            ).apply {
                topMargin = 112
            }
            translationX = nasalShiftPx.toFloat()
        }
    }

    private fun createTelemetryCard(nasalShiftPx: Int): TextView {
        val cardBg = GradientDrawable().apply {
            cornerRadius = 18f
            setColor(Color.argb(204, 11, 16, 27))
            setStroke(2, Color.argb(140, 0, 229, 255))
        }
        return TextView(this).apply {
            visibility = View.GONE
            background = cardBg
            setTextColor(Color.parseColor("#E0F7FA"))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 10f)
            typeface = Typeface.MONOSPACE
            setPadding(18, 12, 18, 12)
            setLineSpacing(2f, 1.08f)
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.WRAP_CONTENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.TOP or Gravity.START
            ).apply {
                topMargin = 58
                leftMargin = 18
            }
            translationX = nasalShiftPx.toFloat()
        }
    }

    private fun addTopControlBar(root: FrameLayout) {
        val bar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }

        drawToggleBtn = makeQuickButton("\u270F Vẽ: TẮT") {
            cycleDrawMode(showPopup = false)
        }
        val clearBtn = makeQuickButton("\uD83D\uDDD1 Xóa") {
            NativeBridge.nativeClearDrawings()
            showStereoBanner("\uD83E\uDDF9  Đã xóa nét vẽ", "#FF5252", 1200L)
        }
        checkUpdateBtn = makeQuickButton("\uD83D\uDD04 Check cập nhật") {
            performUpdateCheck(manual = true)
        }
        installUpdateBtn = makeQuickButton("\u2B07 Cập nhật", fillColor = Color.argb(220, 0, 180, 80)) {
            triggerApkDownloadAndInstall()
        }.apply {
            visibility = View.GONE
        }
        telemetryToggleBtn = makeQuickButton("\uD83D\uDCCA Thông số") {
            toggleTelemetryHud()
        }
        val gearBtn = makeQuickButton("\u2699") {
            panelLayout.visibility =
                if (panelLayout.visibility == View.VISIBLE) View.GONE else View.VISIBLE
        }

        val itemParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.WRAP_CONTENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply {
            marginStart = 8
        }

        bar.addView(drawToggleBtn, itemParams)
        bar.addView(clearBtn, itemParams)
        bar.addView(checkUpdateBtn, itemParams)
        bar.addView(installUpdateBtn, itemParams)
        bar.addView(telemetryToggleBtn, itemParams)
        bar.addView(gearBtn, itemParams)

        val barParams = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT,
            FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.TOP or Gravity.END
        ).apply {
            topMargin = 14
            marginEnd = 16
        }
        root.addView(bar, barParams)
    }

    private fun makeQuickButton(
        label: String,
        fillColor: Int = Color.argb(175, 20, 24, 34),
        onClick: () -> Unit
    ): Button {
        val bg = GradientDrawable().apply {
            cornerRadius = 20f
            setColor(fillColor)
            setStroke(2, Color.argb(130, 255, 255, 255))
        }
        return Button(this).apply {
            text = label
            isAllCaps = false
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 11.5f)
            setTextColor(Color.WHITE)
            background = bg
            minHeight = 0
            minimumHeight = 0
            minWidth = 0
            minimumWidth = 0
            setPadding(18, 10, 18, 10)
            setOnClickListener { onClick() }
        }
    }

    private fun performUpdateCheck(manual: Boolean) {
        if (appUpdater.isDownloading) return
        if (manual) {
            checkUpdateBtn.text = "\uD83D\uDD04 Đang check..."
            checkUpdateBtn.isEnabled = false
        }
        appUpdater.checkForUpdate { res ->
            checkUpdateBtn.text = "\uD83D\uDD04 Check cập nhật"
            checkUpdateBtn.isEnabled = true

            if (res.error != null) {
                if (manual) {
                    showStereoBanner("\u26A0\uFE0F  Lỗi kiểm tra cập nhật: ${res.error}", "#FF7043", 2200L)
                }
                return@checkForUpdate
            }

            if (res.hasUpdate) {
                availableUpdate = res
                val tag = if (res.remoteBuildNumber > 0) "#${res.remoteBuildNumber}" else res.remoteCommit.take(7)
                installUpdateBtn.text = "\u2B07 Cập nhật $tag"
                installUpdateBtn.visibility = View.VISIBLE
                showStereoBanner(
                    "\u2B07  Có bản cập nhật mới ($tag)! Bấm [Cập nhật] để cài",
                    "#00E676",
                    2600L,
                    strokeWidth = 4
                )
            } else {
                availableUpdate = null
                installUpdateBtn.visibility = View.GONE
                if (manual) {
                    showStereoBanner(
                        "\u2705  Đang dùng bản mới nhất (v${BuildConfig.VERSION_NAME})",
                        "#00E5FF",
                        1900L
                    )
                }
            }
        }
    }

    private fun triggerApkDownloadAndInstall() {
        val upd = availableUpdate ?: return
        if (appUpdater.isDownloading) return

        val started = appUpdater.startDownloadAndInstall(
            activity = this,
            downloadUrl = upd.downloadUrl,
            expectedSize = upd.assetSizeBytes,
            onProgress = { pct, _, _ ->
                installUpdateBtn.isEnabled = false
                installUpdateBtn.text = "\u2B07 Đang tải ${pct}%"
            },
            onReadyToInstall = {
                installUpdateBtn.isEnabled = true
                installUpdateBtn.text = "\u2B07 Cài đặt lại"
                showStereoBanner("\uD83D\uDCE6  Đã tải xong — đang mở cài đặt...", "#00E676", 2200L)
            },
            onError = { errMsg ->
                installUpdateBtn.isEnabled = true
                installUpdateBtn.text = "\u2B07 Thử cập nhật lại"
                showStereoBanner("\u26A0\uFE0F  Lỗi tải APK: $errMsg", "#FF5252", 2400L)
            }
        )

        if (!started) {
            pendingInstallPermissionResume = true
            Toast.makeText(
                this,
                "Hãy bật quyền 'Cài đặt ứng dụng không xác định' rồi quay lại app để tự cài đặt.",
                Toast.LENGTH_LONG
            ).show()
        } else {
            showStereoBanner("\u2B07  Đang tải bản cập nhật...", "#00E5FF", 1600L)
        }
    }

    private fun cycleDrawMode(showPopup: Boolean) {
        drawMode = (drawMode + 1) % 3
        val modeLabel = when (drawMode) {
            1 -> "\u270F Vẽ: 1 NGÓN"
            2 -> "\u270F Vẽ: 5 NGÓN"
            else -> "\u270F Vẽ: TẮT"
        }
        drawToggleBtn.text = modeLabel
        NativeBridge.nativeSetDrawMode(drawMode)
        if (showPopup) {
            val banner = when (drawMode) {
                1 -> "\u270F  Chế độ vẽ: 1 NGÓN TRỎ"
                2 -> "\uD83C\uDF08  Chế độ vẽ: 5 ĐẦU NGÓN TAY"
                else -> "\u270F  Đã TẮT chế độ vẽ tay"
            }
            showStereoBanner(banner, "#00E5FF", 1300L)
        }
    }

    private fun toggleTelemetryHud() {
        telemetryVisible = !telemetryVisible
        val vis = if (telemetryVisible) View.VISIBLE else View.GONE
        leftTelemetryCard.visibility = vis
        rightTelemetryCard.visibility = vis
        telemetryToggleBtn.text = if (telemetryVisible) "\uD83D\uDCCA Ẩn thông số" else "\uD83D\uDCCA Thông số"
        if (telemetryVisible) {
            startTelemetryLoop()
        } else {
            stopTelemetryLoop()
        }
    }

    private fun startTelemetryLoop() {
        stopTelemetryLoop()
        val task = object : Runnable {
            override fun run() {
                if (!telemetryVisible) return
                updateTelemetryCards()
                mainHandler.postDelayed(this, 500L)
            }
        }
        telemetryRunnable = task
        mainHandler.post(task)
    }

    private fun stopTelemetryLoop() {
        telemetryRunnable?.let { mainHandler.removeCallbacks(it) }
        telemetryRunnable = null
    }

    private fun updateTelemetryCards() {
        NativeBridge.nativeGetPerfMetrics(metricsBuf)
        if (cachedGpuName.isEmpty()) {
            cachedGpuName = NativeBridge.nativeGetGpuInfo()
        }

        val renderFps = metricsBuf[0].toInt()
        val frameMs = metricsBuf[1]
        val activeHands = metricsBuf[2].toInt()
        val gestureCode = metricsBuf[3].toInt()
        val strokeSegs = metricsBuf[4].toInt() / 48
        val tipX = metricsBuf[5]
        val tipY = metricsBuf[6]
        val tipZ = metricsBuf[7]

        val batIntent = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val level = batIntent?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = batIntent?.getIntExtra(BatteryManager.EXTRA_SCALE, 100) ?: 100
        val batPct = if (level >= 0 && scale > 0) (level * 100) / scale else -1
        val tempTenths = batIntent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0
        val tempC = tempTenths / 10f

        val rt = Runtime.getRuntime()
        val jvmMb = (rt.totalMemory() - rt.freeMemory()) / (1024L * 1024L)
        val nativeMb = Debug.getNativeHeapAllocatedSize() / (1024L * 1024L)
        val totalRamMb = jvmMb + nativeMb

        val soc = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            Build.SOC_MODEL.ifBlank { Build.HARDWARE }
        } else {
            Build.HARDWARE
        }

        val trackMode = if (isUsingArCore) "ARCore 6DoF" else "Camera2 3DoF"
        val tipText = if (activeHands > 0 && tipZ > 0.05f) {
            String.format(Locale.US, "X=%+.2f Y=%+.2f Z=%.2fm", tipX, tipY, tipZ)
        } else {
            "Chưa thấy tay"
        }
        val gestureName = when (gestureCode) {
            1 -> "Nắm đấm"
            2 -> "Chụm ngón"
            3 -> "Cây súng"
            4 -> "Chữ V"
            5 -> "Xòe tay"
            6 -> "Thích"
            7 -> "Rock"
            8 -> "Chỉ 1 ngón"
            else -> "Không"
        }

        val aiFps = handTracker.aiFps.toInt()
        val inferMs = handTracker.lastInferMs
        val shortSha = BuildConfig.GIT_COMMIT_SHA.take(7)

        val summary = StringBuilder()
            .append("Bản: v${BuildConfig.VERSION_NAME} ($shortSha) | ${Build.MODEL} ($soc)\n")
            .append("GPU: $cachedGpuName\n")
            .append("Màn: ${displayModeW}x${displayModeH}@${displayHz}Hz | Pin: ${batPct}% (${String.format(Locale.US, "%.1f", tempC)}°C) | RAM: ${totalRamMb}MB\n")
            .append(String.format(Locale.US, "Render: %d fps (%.1f ms) | AI Tay (%s): %d fps (%d ms)\n", renderFps, frameMs, handTracker.delegateLabel, aiFps, inferMs))
            .append("Track: $trackMode | Tay ($activeHands): $gestureName | Vẽ: ${strokeSegs} nét\n")
            .append("Ngón Trỏ: $tipText")
            .toString()

        leftTelemetryCard.text = summary
        rightTelemetryCard.text = summary
    }

    private fun showStereoGesturePopup(code: Int) {
        val (label, accentHex, durationMs) = when (code) {
            1 -> Triple("\u270A  Phát hiện: NẮM ĐẤM", "#FF7043", 1800L)
            2 -> Triple("\uD83E\uDD0F  Phát hiện: CHỤM NGÓN CÁI", "#00E5FF", 1800L)
            3 -> Triple("\uD83D\uDC49  Phát hiện: HÌNH CÂY SÚNG", "#FFD54F", 1800L)
            4 -> Triple("\u270C  Phát hiện: CHỮ V", "#69F0AE", 1800L)
            6 -> Triple("\uD83D\uDC4D  Phát hiện: THÍCH (LIKE)", "#40C4FF", 1800L)
            7 -> Triple("\uD83E\uDD18  Phát hiện: ROCK / SPIDER", "#CE93D8", 1800L)
            8 -> Triple("\u261D  Phát hiện: CHỈ 1 NGÓN", "#80CBC4", 1600L)
            101 -> Triple("\u26A1 COMBO [Chụm \u2794 Nắm]: Hút / Đẩy khối 3D", "#FFAB40", 2400L)
            102 -> Triple("\u26A1 COMBO [Nắm \u2794 Xòe]: Sóng năng lượng!", "#E040FB", 2400L)
            103 -> Triple("\u26A1 COMBO [Chụm \u2794 Súng]: Đổi màu khối 3D!", "#FFD740", 2400L)
            104 -> Triple("\u26A1 COMBO [Chữ V \u2794 Nắm]: Đã xóa nét vẽ!", "#FF5252", 2400L)
            105 -> Triple("\u26A1 COMBO [Rock \u2794 Xòe]: Đã định tâm khối 3D!", "#00E676", 2400L)
            else -> return
        }
        showStereoBanner(label, accentHex, durationMs, strokeWidth = if (code >= 100) 5 else 3)
    }

    private fun showStereoBanner(label: String, accentHex: String, durationMs: Long, strokeWidth: Int = 3) {
        val accentColor = Color.parseColor(accentHex)
        hidePopupRunnable?.let { mainHandler.removeCallbacks(it) }

        for (pill in arrayOf(leftGesturePopup, rightGesturePopup)) {
            pill.animate().cancel()
            pill.text = label
            (pill.background as? GradientDrawable)?.setStroke(strokeWidth, accentColor)
            pill.alpha = 0f
            pill.scaleX = 0.88f
            pill.scaleY = 0.88f
            pill.visibility = View.VISIBLE
            pill.animate()
                .alpha(1f)
                .scaleX(1f)
                .scaleY(1f)
                .setDuration(140L)
                .start()
        }

        val hideTask = Runnable {
            for (pill in arrayOf(leftGesturePopup, rightGesturePopup)) {
                pill.animate()
                    .alpha(0f)
                    .setDuration(220L)
                    .withEndAction { pill.visibility = View.GONE }
                    .start()
            }
        }
        hidePopupRunnable = hideTask
        mainHandler.postDelayed(hideTask, durationMs)
    }

    private fun addCalibrationOverlay(root: FrameLayout) {
        panelLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            visibility = View.GONE
            setBackgroundColor(Color.argb(232, 16, 18, 24))
            setPadding(32, 24, 32, 24)
        }

        val fovLabel = TextView(this).apply {
            setTextColor(Color.WHITE)
            text = "Khung Camera (chống chóng mặt): $fovPct%"
        }
        val fovBar = SeekBar(this).apply {
            max = 50
            progress = fovPct - 50
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                    fovPct = 50 + p
                    fovLabel.text = "Khung Camera (chống chóng mặt): $fovPct%"
                    saveCalibration()
                }
                override fun onStartTrackingTouch(sb: SeekBar?) {}
                override fun onStopTrackingTouch(sb: SeekBar?) {}
            })
        }

        val ipdLabel = TextView(this).apply {
            setTextColor(Color.WHITE)
            setPadding(0, 16, 0, 0)
            text = "Khoảng cách 2 mắt (IPD): $ipdMm mm"
        }
        val ipdBar = SeekBar(this).apply {
            max = 16
            progress = ipdMm - 56
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                    ipdMm = 56 + p
                    ipdLabel.text = "Khoảng cách 2 mắt (IPD): $ipdMm mm"
                    saveCalibration()
                }
                override fun onStartTrackingTouch(sb: SeekBar?) {}
                override fun onStopTrackingTouch(sb: SeekBar?) {}
            })
        }

        val recenterBtn = Button(this).apply {
            text = "Đặt lại khối 3D sau 3 giây (Nhét vào hộp VR)"
            setOnClickListener {
                panelLayout.visibility = View.GONE
                scheduleDelayedRecenter(this)
            }
        }
        val btnParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply {
            topMargin = 20
        }

        val tipText = TextView(this).apply {
            setTextColor(Color.LTGRAY)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 11.5f)
            setPadding(0, 12, 0, 0)
            text = "Mẹo hộp VR: Volume+ = Đặt lại khối 3D | Volume- = Đổi chế độ Vẽ tay (Tắt / 1 ngón / 5 ngón)\nCombo: Chụm➔Nắm (Hút/Đẩy), Nắm➔Xòe (Sóng), Chụm➔Súng (Đổi màu), V➔Nắm (Xóa nét vẽ)"
        }

        panelLayout.addView(fovLabel)
        panelLayout.addView(fovBar)
        panelLayout.addView(ipdLabel)
        panelLayout.addView(ipdBar)
        panelLayout.addView(recenterBtn, btnParams)
        panelLayout.addView(tipText)

        val panelParams = FrameLayout.LayoutParams(
            820,
            FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.CENTER
        )
        root.addView(panelLayout, panelParams)
    }

    private fun scheduleDelayedRecenter(btn: Button) {
        recenterRunnable?.let { mainHandler.removeCallbacks(it) }
        Toast.makeText(this, "Hãy đặt máy vào hộp VR — định tâm sau 3 giây...", Toast.LENGTH_SHORT).show()
        val task = Runnable {
            NativeBridge.nativeRecenter()
            btn.text = "Đặt lại khối 3D sau 3 giây (Nhét vào hộp VR)"
        }
        recenterRunnable = task
        mainHandler.postDelayed(task, 3000L)
    }

    override fun onResume() {
        super.onResume()
        applyImmersiveFullscreen()
        updateDisplayRotation()
        rotSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST)
        }
        glView.onResume()
        if (hasCameraPermission()) {
            startTrackingAndCamera()
        }
        if (telemetryVisible) {
            startTelemetryLoop()
        }
        if (pendingInstallPermissionResume) {
            pendingInstallPermissionResume = false
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O || packageManager.canRequestPackageInstalls()) {
                triggerApkDownloadAndInstall()
            }
        }
    }

    override fun onPause() {
        recenterRunnable?.let { mainHandler.removeCallbacks(it) }
        hidePopupRunnable?.let { mainHandler.removeCallbacks(it) }
        stopTelemetryLoop()
        sensorManager.unregisterListener(this)
        stopTrackingAndCamera()
        glView.onPause()
        super.onPause()
    }

    override fun onDestroy() {
        handTracker.close()
        appUpdater.close()
        synchronized(sessionLock) {
            arSession?.close()
            arSession = null
        }
        super.onDestroy()
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) {
            applyImmersiveFullscreen()
        }
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        if (keyCode == KeyEvent.KEYCODE_VOLUME_UP) {
            NativeBridge.nativeRecenter()
            showStereoBanner("\uD83E\uDDED  Đã đặt lại vị trí khối 3D", "#00E5FF", 1300L)
            return true
        }
        if (keyCode == KeyEvent.KEYCODE_VOLUME_DOWN) {
            cycleDrawMode(showPopup = true)
            return true
        }
        return super.onKeyDown(keyCode, event)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_CAMERA &&
            grantResults.isNotEmpty() &&
            grantResults[0] == PackageManager.PERMISSION_GRANTED
        ) {
            startTrackingAndCamera()
        } else if (requestCode == REQ_CAMERA) {
            Toast.makeText(this, "Cần quyền Camera để nhìn xuyên thấu.", Toast.LENGTH_LONG).show()
        }
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type == Sensor.TYPE_GAME_ROTATION_VECTOR ||
            event.sensor.type == Sensor.TYPE_ROTATION_VECTOR
        ) {
            SensorManager.getQuaternionFromVector(quatBuf, event.values)
            NativeBridge.nativeOnGyroRotation(
                quatBuf[1],
                quatBuf[2],
                quatBuf[3],
                quatBuf[0],
                displayRotation
            )
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun hasCameraPermission(): Boolean {
        return checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
    }

    private fun updateDisplayRotation() {
        val rot = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            display?.rotation ?: Surface.ROTATION_90
        } else {
            @Suppress("DEPRECATION")
            windowManager.defaultDisplay.rotation
        }
        displayRotation = rot
        viewportDirty = true
    }

    private fun startTrackingAndCamera() {
        synchronized(sessionLock) {
            if (!arInitAttempted) {
                arInitAttempted = true
                try {
                    val session = Session(this)
                    val filter = CameraConfigFilter(session).apply {
                        targetFps = EnumSet.of(CameraConfig.TargetFps.TARGET_FPS_30)
                    }
                    val configs = session.getSupportedCameraConfigs(filter)
                    if (configs.isNotEmpty()) {
                        val preferred = configs
                            .filter { it.imageSize.height <= 480 && it.textureSize.height >= 1080 }
                            .maxByOrNull { it.textureSize.height }
                            ?: configs.filter { it.imageSize.height <= 720 }.maxByOrNull { it.textureSize.height }
                            ?: configs[0]
                        session.cameraConfig = preferred
                        Log.i(TAG, "ARCore CameraConfig: cpu=${preferred.imageSize}, gpu=${preferred.textureSize}")
                    }
                    val config = Config(session).apply {
                        focusMode = Config.FocusMode.AUTO
                        planeFindingMode = Config.PlaneFindingMode.DISABLED
                        lightEstimationMode = Config.LightEstimationMode.DISABLED
                        depthMode = Config.DepthMode.DISABLED
                        updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
                    }
                    session.configure(config)
                    arSession = session
                    arNeedsTextureBind = true
                    isUsingArCore = true
                    Log.i(TAG, "ARCore 6DoF initialized (depth disabled for max speed)")
                } catch (t: Throwable) {
                    Log.w(TAG, "ARCore unavailable, falling back to Camera2 + 120Hz 3DoF: ${t.message}")
                    arSession = null
                    isUsingArCore = false
                }
            }

            val session = arSession
            if (session != null) {
                try {
                    session.resume()
                    arNeedsTextureBind = true
                    viewportDirty = true
                    isUsingArCore = true
                    return
                } catch (t: Throwable) {
                    Log.w(TAG, "ARCore resume failed, falling back to Camera2: ${t.message}")
                    session.close()
                    arSession = null
                    isUsingArCore = false
                }
            }
        }
        startFallbackCameraIfReady()
    }

    private fun stopTrackingAndCamera() {
        synchronized(sessionLock) {
            try {
                arSession?.pause()
            } catch (t: Throwable) {
                Log.w(TAG, "ARCore pause warning: ${t.message}")
            }
            closeFallbackCameraLocked()
        }
    }

    @SuppressLint("MissingPermission")
    private fun startFallbackCameraIfReady() {
        if (!hasCameraPermission()) return
        val texId = cameraTexId
        if (texId == 0) return

        synchronized(sessionLock) {
            if (arSession != null || fallbackCameraActive) return
            fallbackCameraActive = true
            isUsingArCore = false
        }

        val thread = HandlerThread("Camera2Fallback").apply { start() }
        val handler = Handler(thread.looper)
        cameraThread = thread
        cameraHandler = handler

        val camMgr = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val camId = camMgr.cameraIdList.firstOrNull { id ->
            camMgr.getCameraCharacteristics(id)
                .get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
        } ?: camMgr.cameraIdList.firstOrNull() ?: return

        val chars = camMgr.getCameraCharacteristics(camId)
        val physSize = chars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
        val focals = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
        val streamMap = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        val sizes = streamMap?.getOutputSizes(SurfaceTexture::class.java)
        val chosenSize = sizes
            ?.filter { it.width in 960..1920 && it.height in 540..1080 }
            ?.minByOrNull { abs(it.height - 720) }
            ?: Size(1280, 720)

        if (physSize != null && focals != null && focals.isNotEmpty() && focals[0] > 0.1f) {
            val f = focals[0]
            fallbackTanHalfH = ((physSize.width * 0.5f) / f).coerceIn(0.3f, 1.5f)
            fallbackTanHalfV = fallbackTanHalfH * (chosenSize.height.toFloat() / chosenSize.width.toFloat())
        }

        val st = SurfaceTexture(texId).apply {
            setDefaultBufferSize(chosenSize.width, chosenSize.height)
            setOnFrameAvailableListener({ fallbackFrameAvailable.set(true) }, handler)
        }
        val surface = Surface(st)

        val previewAspect = chosenSize.width.toFloat() / chosenSize.height.toFloat()
        val yuvSizes = streamMap?.getOutputSizes(ImageFormat.YUV_420_888)
        val yuvSize = yuvSizes
            ?.filter {
                it.width in 480..1280 &&
                    it.height in 320..720 &&
                    abs((it.width.toFloat() / it.height.toFloat()) - previewAspect) < 0.08f
            }
            ?.minByOrNull { abs(it.height - 480) }
            ?: yuvSizes
                ?.filter { it.width in 480..1280 && it.height in 320..720 }
                ?.minByOrNull { abs(it.height - 480) }
            ?: Size(640, 480)

        val ir = ImageReader.newInstance(yuvSize.width, yuvSize.height, ImageFormat.YUV_420_888, 2).apply {
            setOnImageAvailableListener({ reader ->
                val img = try {
                    reader.acquireLatestImage()
                } catch (_: Throwable) {
                    null
                } ?: return@setOnImageAvailableListener
                try {
                    if (!handTracker.isBusy.get()) {
                        val rot = displayRotation
                        if (rot == Surface.ROTATION_270) {
                            handTracker.submitYuvFrame(img, rot, 1f, -1f, -1f, -1f, 1f, 1f)
                        } else {
                            handTracker.submitYuvFrame(img, rot, -1f, 1f, 1f, 1f, -1f, -1f)
                        }
                    }
                } finally {
                    img.close()
                }
            }, handler)
        }

        synchronized(sessionLock) {
            fallbackSurfaceTexture = st
            fallbackPreviewSurface = surface
            fallbackImageReader = ir
        }

        try {
            camMgr.openCamera(camId, object : CameraDevice.StateCallback() {
                override fun onOpened(device: CameraDevice) {
                    synchronized(sessionLock) {
                        if (!fallbackCameraActive) {
                            device.close()
                            return
                        }
                        cameraDevice = device
                    }
                    startCamera2Session(device, surface, ir.surface, handler)
                }

                override fun onDisconnected(device: CameraDevice) {
                    device.close()
                }

                override fun onError(device: CameraDevice, error: Int) {
                    device.close()
                }
            }, handler)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to open fallback camera: ${e.message}")
        }
    }

    private fun startCamera2Session(
        device: CameraDevice,
        previewSurface: Surface,
        analysisSurface: Surface?,
        handler: Handler
    ) {
        try {
            val targets = if (analysisSurface != null) {
                listOf(previewSurface, analysisSurface)
            } else {
                listOf(previewSurface)
            }
            val req = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                for (t in targets) addTarget(t)
                set(
                    CaptureRequest.CONTROL_AF_MODE,
                    CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO
                )
            }
            @Suppress("DEPRECATION")
            device.createCaptureSession(
                targets,
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        synchronized(sessionLock) {
                            if (!fallbackCameraActive) {
                                session.close()
                                return
                            }
                            captureSession = session
                        }
                        try {
                            session.setRepeatingRequest(req.build(), null, handler)
                        } catch (e: Exception) {
                            Log.e(TAG, "Camera2 preview failed: ${e.message}")
                        }
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        if (analysisSurface != null) {
                            Log.w(TAG, "Camera2 dual-surface failed, retrying preview-only")
                            startCamera2Session(device, previewSurface, null, handler)
                        } else {
                            Log.e(TAG, "Camera2 session configure failed")
                        }
                    }
                },
                handler
            )
        } catch (e: Exception) {
            if (analysisSurface != null) {
                startCamera2Session(device, previewSurface, null, handler)
            } else {
                Log.e(TAG, "Camera2 createCaptureSession failed: ${e.message}")
            }
        }
    }

    private fun closeFallbackCameraLocked() {
        fallbackCameraActive = false
        try {
            captureSession?.close()
        } catch (_: Throwable) {}
        captureSession = null

        try {
            cameraDevice?.close()
        } catch (_: Throwable) {}
        cameraDevice = null

        try {
            fallbackImageReader?.close()
        } catch (_: Throwable) {}
        fallbackImageReader = null

        try {
            fallbackPreviewSurface?.release()
        } catch (_: Throwable) {}
        fallbackPreviewSurface = null

        try {
            fallbackSurfaceTexture?.release()
        } catch (_: Throwable) {}
        fallbackSurfaceTexture = null

        cameraThread?.quitSafely()
        cameraThread = null
        cameraHandler = null
    }

    private inner class ArVrGlRenderer : GLSurfaceView.Renderer {
        override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
            val texId = NativeBridge.nativeOnSurfaceCreated()
            cameraTexId = texId
            cachedGpuName = NativeBridge.nativeGetGpuInfo()
            synchronized(sessionLock) {
                arNeedsTextureBind = true
            }
            if (arSession == null && hasCameraPermission()) {
                mainHandler.post { startFallbackCameraIfReady() }
            }
        }

        override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
            viewWidth = width
            viewHeight = height
            viewportDirty = true
            updateDisplayRotation()
            NativeBridge.nativeOnSurfaceChanged(width, height)
        }

        override fun onDrawFrame(gl: GL10?) {
            val nowNs = System.nanoTime()

            synchronized(sessionLock) {
                val session = arSession
                if (session != null && cameraTexId != 0) {
                    if (arNeedsTextureBind) {
                        session.setCameraTextureName(cameraTexId)
                        arNeedsTextureBind = false
                    }
                    if (viewportDirty) {
                        session.setDisplayGeometry(displayRotation, viewWidth, viewHeight)
                        viewportDirty = false
                    }
                    if (nowNs - lastArUpdateNs >= 31_000_000L) {
                        lastArUpdateNs = nowNs
                        try {
                            val frame = session.update()
                            val ts = frame.timestamp
                            if (ts != lastArTimestampNs) {
                                lastArTimestampNs = ts
                                val cam = frame.camera
                                val valid = cam.trackingState == TrackingState.TRACKING
                                val pose = cam.displayOrientedPose

                                cam.getProjectionMatrix(arProjMat, 0, 0.05f, 50f)
                                val tanH = if (abs(arProjMat[0]) > 0.01f) 1f / abs(arProjMat[0]) else 0.67f
                                val tanV = if (abs(arProjMat[5]) > 0.01f) 1f / abs(arProjMat[5]) else 0.50f

                                ndcQuadBuf.position(0)
                                uvOutBuf.position(0)
                                frame.transformCoordinates2d(
                                    Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES,
                                    ndcQuadBuf,
                                    Coordinates2d.TEXTURE_NORMALIZED,
                                    uvOutBuf
                                )
                                uvOutBuf.position(0)
                                uvOutBuf.get(uvArray)

                                NativeBridge.nativeOnArFrame(
                                    pose.tx(),
                                    pose.ty(),
                                    pose.tz(),
                                    pose.qx(),
                                    pose.qy(),
                                    pose.qz(),
                                    pose.qw(),
                                    tanH,
                                    tanV,
                                    uvArray,
                                    valid
                                )

                                // Fast 480p GPU MediaPipe Hand Tracking
                                if (!handTracker.isBusy.get()) {
                                    try {
                                        frame.transformCoordinates2d(
                                            Coordinates2d.IMAGE_NORMALIZED,
                                            imgBasis,
                                            Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES,
                                            ndcBasis
                                        )
                                        val cpuImg = frame.acquireCameraImage()
                                        try {
                                            handTracker.submitYuvFrame(
                                                cpuImg,
                                                displayRotation,
                                                ndcBasis[0], ndcBasis[1],
                                                ndcBasis[2], ndcBasis[3],
                                                ndcBasis[4], ndcBasis[5]
                                            )
                                        } finally {
                                            cpuImg.close()
                                        }
                                    } catch (_: Throwable) {
                                    }
                                }
                            }
                        } catch (t: Throwable) {
                            Log.w(TAG, "ARCore update exception, switching to Camera2 fallback: ${t.message}")
                            try {
                                session.close()
                            } catch (_: Throwable) {}
                            arSession = null
                            isUsingArCore = false
                            mainHandler.post { startFallbackCameraIfReady() }
                        }
                    }
                } else if (fallbackCameraActive && fallbackFrameAvailable.compareAndSet(true, false)) {
                    val st = fallbackSurfaceTexture
                    if (st != null) {
                        try {
                            st.updateTexImage()
                            st.getTransformMatrix(fallbackTexMtx)
                            computeFallbackUvs(fallbackTexMtx, displayRotation, uvArray)
                            NativeBridge.nativeOnFallbackCameraFrame(
                                fallbackTanHalfH,
                                fallbackTanHalfV,
                                uvArray
                            )
                        } catch (t: Throwable) {
                            Log.w(TAG, "Fallback texture update warning: ${t.message}")
                        }
                    }
                }
            }

            NativeBridge.nativeDrawFrame()
        }
    }

    private fun computeFallbackUvs(mtx: FloatArray, rot: Int, out: FloatArray) {
        val flip = rot == Surface.ROTATION_270
        val corners = if (!flip) {
            floatArrayOf(0f, 0f, 1f, 0f, 0f, 1f, 1f, 1f)
        } else {
            floatArrayOf(1f, 1f, 0f, 1f, 1f, 0f, 0f, 0f)
        }
        for (i in 0 until 4) {
            val s = corners[i * 2]
            val t = corners[i * 2 + 1]
            out[i * 2] = mtx[0] * s + mtx[4] * t + mtx[12]
            out[i * 2 + 1] = mtx[1] * s + mtx[5] * t + mtx[13]
        }
    }

    private class FastEglConfigChooser : GLSurfaceView.EGLConfigChooser {
        override fun chooseConfig(egl: EGL10, display: EGLDisplay): EGLConfig {
            val numConfig = IntArray(1)
            val configs = arrayOfNulls<EGLConfig>(1)

            val fastAttribs = intArrayOf(
                EGL10.EGL_RED_SIZE, 8,
                EGL10.EGL_GREEN_SIZE, 8,
                EGL10.EGL_BLUE_SIZE, 8,
                EGL10.EGL_DEPTH_SIZE, 16,
                EGL10.EGL_RENDERABLE_TYPE, 4,
                EGL10.EGL_NONE
            )
            if (egl.eglChooseConfig(display, fastAttribs, configs, 1, numConfig) &&
                numConfig[0] > 0 && configs[0] != null
            ) {
                return configs[0]!!
            }
            throw IllegalStateException("No compatible EGLConfig found")
        }
    }
}
