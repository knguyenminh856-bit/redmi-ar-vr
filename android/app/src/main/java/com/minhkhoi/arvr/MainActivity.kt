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
import android.media.Image
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
import android.view.SurfaceHolder
import android.view.View
import android.view.WindowInsets
import android.view.WindowInsetsController
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ScrollView
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
import kotlin.math.max
import kotlin.math.min

class MainActivity : Activity(), SensorEventListener {

    companion object {
        private const val TAG = "RedmiARVR"
        private const val REQ_CAMERA = 1001
        private const val PREFS_NAME = "vr_calibration"
        private const val KEY_FOV_PCT_V2 = "fov_pct_v2"
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
    private var fallbackReqBuilder: CaptureRequest.Builder? = null
    private var fallbackSurfaceTexture: SurfaceTexture? = null
    private var fallbackPreviewSurface: Surface? = null
    private var fallbackImageReader: ImageReader? = null
    private val fallbackFrameAvailable = AtomicBoolean(false)
    private var fallbackCameraActive = false
    private var fallbackTanHalfH = 0.67f
    private var fallbackTanHalfV = 0.50f
    private val fallbackTexMtx = FloatArray(16)

    private lateinit var handTracker: HandTrackerHelper
    private lateinit var roomStorage: RoomMapStorage
    private lateinit var appUpdater: AppUpdater
    private var autoSaveRunnable: Runnable? = null
    private var autoCacheLoadedAttempted = false
    private val imgBasis = floatArrayOf(0f, 0f, 1f, 0f, 0f, 1f)
    private val ndcBasis = FloatArray(6)
    private val fallbackBasis90 = floatArrayOf(-1f, 1f, 1f, 1f, -1f, -1f)
    private val fallbackBasis270 = floatArrayOf(1f, -1f, -1f, -1f, 1f, 1f)

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
    private var fovPct = 100
    private var ipdMm = 64
    @Volatile
    private var stereoMode = false // false=1 Mắt (Full màn zero-IPD), true=2 Mắt (VR Stereo)
    @Volatile
    private var meshOcclusionEnabled = true // Che khuất bằng Hardware Z-Buffer từ lưới TSDF 3D tĩnh
    @Volatile
    private var depthSupportLevel = 0 // 2=AUTOMATIC Dense Depth, 1=RAW_DEPTH_ONLY, 0=SLAM
    @Volatile
    private var scanTurbo = true // true=⚡ Quét mạnh (9Hz liên tục), false=🔒 Khóa (2Hz chỉ đổi khi thay đổi)
    @Volatile
    private var torchEnabled = false // Đèn flash LED chiếu sáng phòng khi quét chi tiết
    @Volatile
    private var drawMode = 0 // 0=Off, 1=1 Finger, 2=5 Fingers
    @Volatile
    private var handMode = 0 // 0=Tắt (mặc định tối ưu 3D), 1=Tự động, 2=Nhanh
    @Volatile
    private var roomScanMode = 4 // 4=Lưới △ (Thích ứng 2.5cm + Gộp phẳng + Quét sàn), 3=Ẩn lưới, 0=TẮT
    private var telemetryVisible = false
    private var availableUpdate: UpdateCheckResult? = null
    private var pendingInstallPermissionResume = false

    private lateinit var panelLayout: LinearLayout
    private lateinit var leftGesturePopup: TextView
    private lateinit var rightGesturePopup: TextView
    private lateinit var leftReticleHud: TextView
    private lateinit var rightReticleHud: TextView
    private var hidePopupRunnable: Runnable? = null
    private lateinit var leftRoomPane: FrameLayout
    private lateinit var rightRoomPane: FrameLayout
    private val pointCloudBuf = FloatArray(512 * 4)
    private val depthFocalBuf = FloatArray(2)
    private val depthPpBuf = FloatArray(2)
    private val depthDimsBuf = IntArray(2)
    private var lastPointCloudNs = 0L
    private var lastRawDepthNs = 0L
    private var lastDepthTimestampNs = 0L
    private var lastRelocIngestNs = 0L
    private var blurSettleUntilNs = 0L
    private var hasPrevArPose = false
    private var prevArPosX = 0f
    private var prevArPosY = 0f
    private var prevArPosZ = 0f
    private var prevArQx = 0f
    private var prevArQy = 0f
    private var prevArQz = 0f
    private var prevArQw = 1f
    private var prevArTsNs = 0L
    @Volatile
    private var camAngSpeedRadS = 0f
    @Volatile
    private var camLinSpeedMS = 0f
    @Volatile
    private var blurSkipActive = false
    @Volatile
    private var relocPending = false
    private var reticleHudRunnable: Runnable? = null
    private lateinit var leftTelemetryCard: TextView
    private lateinit var rightTelemetryCard: TextView
    private lateinit var eyeModeBtn: Button
    private lateinit var drawToggleBtn: Button
    private lateinit var handModeBtn: Button
    private lateinit var roomScanToggleBtn: Button
    private lateinit var scanTurboBtn: Button
    private lateinit var torchBtn: Button
    private lateinit var checkUpdateBtn: Button
    private lateinit var installUpdateBtn: Button
    private lateinit var telemetryToggleBtn: Button
    private lateinit var roomPanelLayout: LinearLayout
    private lateinit var roomStatusText: TextView
    private lateinit var roomListContainer: LinearLayout
    private lateinit var relocBar: LinearLayout
    private lateinit var relocStatusLabel: TextView
    private val metricsBuf = FloatArray(26)
    private var cachedGpuName = ""
    private var telemetryRunnable: Runnable? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        NativeBridge.nativeInit()
        handTracker = HandTrackerHelper(this, mainHandler) { code ->
            showStereoGesturePopup(code)
        }
        handTracker.setMode(handMode)
        roomStorage = RoomMapStorage(this, mainHandler)
        appUpdater = AppUpdater(mainHandler)

        loadCalibration()
        NativeBridge.nativeSetStereoMode(stereoMode)
        NativeBridge.nativeSetMeshOcclusion(meshOcclusionEnabled)
        NativeBridge.nativeSetScanTurbo(scanTurbo)
        NativeBridge.nativeSetDrawMode(drawMode)
        NativeBridge.nativeSetRoomScanMode(roomScanMode)
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
        glView.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) {
                applySurfaceRefreshRate(holder.surface)
            }

            override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
                applySurfaceRefreshRate(holder.surface)
            }

            override fun surfaceDestroyed(holder: SurfaceHolder) {}
        })
        root.addView(
            glView,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        )

        addStereoHudAndPopupOverlay(root)
        addTopControlBar(root)
        addRoomManagerOverlay(root)
        addCalibrationOverlay(root)
        setContentView(root)
        applyImmersiveFullscreen()

        if (!hasCameraPermission()) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), REQ_CAMERA)
        }

        mainHandler.postDelayed({ performUpdateCheck(manual = false) }, 2500L)
        mainHandler.postDelayed({ maybeAutoLoadRoomCache() }, 900L)
    }

    private fun dp(v: Float): Int =
        TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, resources.displayMetrics).toInt()

    private fun applySurfaceRefreshRate(surface: Surface) {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                surface.setFrameRate(
                    120f,
                    Surface.FRAME_RATE_COMPATIBILITY_FIXED_SOURCE,
                    Surface.CHANGE_FRAME_RATE_ONLY_IF_SEAMLESS
                )
            } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                surface.setFrameRate(120f, Surface.FRAME_RATE_COMPATIBILITY_FIXED_SOURCE)
            }
        } catch (_: Throwable) {
        }
    }

    private fun loadCalibration() {
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        fovPct = prefs.getInt(KEY_FOV_PCT_V2, 100).coerceIn(55, 105)
        ipdMm = prefs.getInt(KEY_IPD_MM, 64).coerceIn(56, 72)
        NativeBridge.nativeSetCalibration(fovPct / 100f, ipdMm / 1000f)
    }

    private fun saveCalibration() {
        getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putInt(KEY_FOV_PCT_V2, fovPct)
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

        displayModeW = max(bestMode.physicalWidth, bestMode.physicalHeight)
        displayModeH = min(bestMode.physicalWidth, bestMode.physicalHeight)
        displayHz = bestMode.refreshRate.toInt()

        val attrs = window.attributes
        if (bestMode.modeId != curMode.modeId) {
            attrs.preferredDisplayModeId = bestMode.modeId
        }
        attrs.preferredRefreshRate = bestMode.refreshRate
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            attrs.layoutInDisplayCutoutMode =
                WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_ALWAYS
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            attrs.layoutInDisplayCutoutMode =
                WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
        }
        window.attributes = attrs
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
        leftReticleHud = createReticleHudPill(nasalShiftPx = 4)
        rightReticleHud = createReticleHudPill(nasalShiftPx = -4)

        val leftPane = FrameLayout(this).apply {
            isClickable = false
            isFocusable = false
            addView(leftTelemetryCard)
            addView(leftGesturePopup)
            addView(leftReticleHud)
        }
        val rightPane = FrameLayout(this).apply {
            isClickable = false
            isFocusable = false
            addView(rightTelemetryCard)
            addView(rightGesturePopup)
            addView(rightReticleHud)
        }
        leftRoomPane = leftPane
        rightRoomPane = rightPane

        stereoRow.addView(
            leftPane,
            LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, 1f)
        )
        stereoRow.addView(
            rightPane,
            LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, 1f)
        )

        root.addView(
            stereoRow,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        )
        applyStereoUiLayout()
    }

    private fun createReticleHudPill(nasalShiftPx: Int): TextView {
        val bg = GradientDrawable().apply {
            cornerRadius = 26f
            setColor(Color.argb(182, 10, 16, 26))
            setStroke(2, Color.argb(185, 46, 230, 230))
        }
        return TextView(this).apply {
            visibility = View.GONE
            background = bg
            setTextColor(Color.parseColor("#E0F7FA"))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 11.5f)
            typeface = Typeface.DEFAULT_BOLD
            setPadding(dp(12f), dp(5f), dp(12f), dp(5f))
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.WRAP_CONTENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL
            ).apply {
                bottomMargin = dp(12f)
            }
            translationX = nasalShiftPx.toFloat()
        }
    }

    private fun applyStereoUiLayout() {
        val leftLp = leftRoomPane.layoutParams as LinearLayout.LayoutParams
        val rightLp = rightRoomPane.layoutParams as LinearLayout.LayoutParams
        if (stereoMode) {
            leftLp.weight = 1f
            rightLp.weight = 1f
            rightRoomPane.visibility = View.VISIBLE
            leftGesturePopup.translationX = 8f
            leftTelemetryCard.translationX = 6f
            leftReticleHud.translationX = 4f
            leftTelemetryCard.setTextSize(TypedValue.COMPLEX_UNIT_SP, 7.5f)
            rightTelemetryCard.setTextSize(TypedValue.COMPLEX_UNIT_SP, 7.5f)
        } else {
            leftLp.weight = 2f
            rightLp.weight = 0f
            rightRoomPane.visibility = View.GONE
            leftGesturePopup.translationX = 0f
            leftTelemetryCard.translationX = 0f
            leftReticleHud.translationX = 0f
            leftTelemetryCard.setTextSize(TypedValue.COMPLEX_UNIT_SP, 9.0f)
        }
        leftRoomPane.layoutParams = leftLp
        rightRoomPane.layoutParams = rightLp
        rightTelemetryCard.visibility = if (stereoMode && telemetryVisible) View.VISIBLE else View.GONE
    }

    private fun createPopupPill(nasalShiftPx: Int): TextView {
        val pillBg = GradientDrawable().apply {
            cornerRadius = 32f
            setColor(Color.argb(232, 14, 20, 34))
            setStroke(3, Color.parseColor("#00E5FF"))
        }
        return TextView(this).apply {
            visibility = View.GONE
            background = pillBg
            setTextColor(Color.WHITE)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12.5f)
            typeface = Typeface.DEFAULT_BOLD
            setPadding(dp(14f), dp(7f), dp(14f), dp(7f))
            gravity = Gravity.CENTER
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.WRAP_CONTENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL
            ).apply {
                bottomMargin = dp(52f)
            }
            translationX = nasalShiftPx.toFloat()
        }
    }

    private fun createTelemetryCard(nasalShiftPx: Int): TextView {
        val cardBg = GradientDrawable().apply {
            cornerRadius = 16f
            setColor(Color.argb(196, 10, 15, 25))
            setStroke(2, Color.argb(145, 0, 229, 255))
        }
        return TextView(this).apply {
            visibility = View.GONE
            background = cardBg
            setTextColor(Color.parseColor("#E0F7FA"))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 9.0f)
            typeface = Typeface.MONOSPACE
            setPadding(dp(8f), dp(5f), dp(8f), dp(5f))
            setLineSpacing(1f, 1.06f)
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.WRAP_CONTENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.TOP or Gravity.START
            ).apply {
                topMargin = dp(54f)
                leftMargin = dp(10f)
            }
            translationX = nasalShiftPx.toFloat()
        }
    }

    private fun addTopControlBar(root: FrameLayout) {
        val bar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }

        roomScanToggleBtn = makeQuickButton(
            roomScanLabel(roomScanMode),
            fillColor = if (roomScanMode == 4) Color.argb(225, 14, 138, 108) else Color.argb(190, 24, 30, 42)
        ) {
            cycleRoomScanMode(showPopup = true)
        }
        scanTurboBtn = makeQuickButton(
            if (scanTurbo) "\u26A1 Quét mạnh" else "\uD83D\uDD12 Khóa",
            fillColor = if (scanTurbo) Color.argb(232, 217, 119, 6) else Color.argb(220, 14, 116, 144)
        ) {
            toggleScanTurbo(showPopup = true)
        }
        torchBtn = makeQuickButton("\uD83D\uDD26 Đèn") {
            toggleTorch(showPopup = true)
        }
        val clearCacheTopBtn = makeQuickButton("\uD83E\uDDF9 Xóa cache", fillColor = Color.argb(215, 165, 58, 28)) {
            clearRoomCacheAndReset()
        }
        val roomMgrBtn = makeQuickButton("\uD83D\uDCBE Phòng", fillColor = Color.argb(215, 36, 70, 128)) {
            toggleRoomManagerPanel()
        }
        checkUpdateBtn = makeQuickButton("\uD83D\uDD04") {
            performUpdateCheck(manual = true)
        }
        installUpdateBtn = makeQuickButton("\u2B07 C.Nhật", fillColor = Color.argb(230, 0, 180, 80)) {
            triggerApkDownloadAndInstall()
        }.apply {
            visibility = View.GONE
        }
        val gearBtn = makeQuickButton("\u2699 Cài đặt", fillColor = Color.argb(205, 42, 48, 64)) {
            roomPanelLayout.visibility = View.GONE
            panelLayout.visibility =
                if (panelLayout.visibility == View.VISIBLE) View.GONE else View.VISIBLE
        }

        val itemParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.WRAP_CONTENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply {
            marginStart = dp(5f)
        }

        bar.addView(roomScanToggleBtn, itemParams)
        bar.addView(scanTurboBtn, itemParams)
        bar.addView(torchBtn, itemParams)
        bar.addView(clearCacheTopBtn, itemParams)
        bar.addView(roomMgrBtn, itemParams)
        bar.addView(checkUpdateBtn, itemParams)
        bar.addView(installUpdateBtn, itemParams)
        bar.addView(gearBtn, itemParams)

        val barParams = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT,
            FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.TOP or Gravity.END
        ).apply {
            topMargin = dp(8f)
            marginEnd = dp(8f)
        }
        root.addView(bar, barParams)

        // Large, thumb-friendly bottom-right physics action dock (Clear | Drop | Throw)
        val actionDock = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val clearPhysBtn = makeQuickButton("\uD83D\uDDD1", fillColor = Color.argb(205, 90, 36, 42)) {
            NativeBridge.nativeClearDrawings()
            NativeBridge.nativeClearPhysicsBodies()
            showStereoBanner("\uD83E\uDDF9  Đã xóa bóng vật lý & nét vẽ", "#FF5252", 1200L)
        }.apply {
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 13.5f)
            minHeight = dp(46f)
            minimumHeight = dp(46f)
            minWidth = dp(46f)
            minimumWidth = dp(46f)
            setPadding(dp(12f), dp(10f), dp(12f), dp(10f))
        }
        val dropBtn = makeQuickButton("\u2B07 Thả", fillColor = Color.argb(230, 16, 136, 148)) {
            NativeBridge.nativeDropPhysicsAtReticle()
        }.apply {
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 13.5f)
            typeface = Typeface.DEFAULT_BOLD
            minHeight = dp(46f)
            minimumHeight = dp(46f)
            setPadding(dp(16f), dp(10f), dp(16f), dp(10f))
        }
        val throwBtn = makeQuickButton("\uD83C\uDFAF Ném", fillColor = Color.argb(230, 192, 98, 22)) {
            NativeBridge.nativeThrowPhysicsAtReticle(2.4f)
        }.apply {
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 13.5f)
            typeface = Typeface.DEFAULT_BOLD
            minHeight = dp(46f)
            minimumHeight = dp(46f)
            setPadding(dp(16f), dp(10f), dp(16f), dp(10f))
        }
        val dockItemParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.WRAP_CONTENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply {
            marginStart = dp(7f)
        }
        actionDock.addView(clearPhysBtn, dockItemParams)
        actionDock.addView(dropBtn, dockItemParams)
        actionDock.addView(throwBtn, dockItemParams)

        val dockParams = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT,
            FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.BOTTOM or Gravity.END
        ).apply {
            bottomMargin = dp(12f)
            marginEnd = dp(12f)
        }
        root.addView(actionDock, dockParams)
    }

    private fun makeQuickButton(
        label: String,
        fillColor: Int = Color.argb(190, 24, 30, 42),
        onClick: () -> Unit
    ): Button {
        val bg = GradientDrawable().apply {
            cornerRadius = 24f
            setColor(fillColor)
            setStroke(2, Color.argb(150, 255, 255, 255))
        }
        return Button(this).apply {
            text = label
            isAllCaps = false
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 11.5f)
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(Color.WHITE)
            background = bg
            minHeight = dp(38f)
            minimumHeight = dp(38f)
            minWidth = dp(40f)
            minimumWidth = dp(40f)
            setPadding(dp(10f), dp(6f), dp(10f), dp(6f))
            setOnClickListener { onClick() }
        }
    }

    private fun handModeLabel(mode: Int): String = when (mode) {
        2 -> "\uD83D\uDD90 AI Tay: NHANH"
        1 -> "\uD83D\uDD90 AI Tay: T.Đ"
        else -> "\uD83D\uDD90 AI Tay: TẮT"
    }

    private fun roomScanLabel(mode: Int): String = when (mode) {
        4 -> "\uD83D\uDD3A Lưới"
        3 -> "\uD83D\uDC41 Ẩn lưới"
        else -> "\u23F9 TẮT"
    }

    private fun hasScannedRoomData(): Boolean {
        NativeBridge.nativeGetPerfMetrics(metricsBuf)
        val totalTris = metricsBuf[17].toInt()
        val cachedPlanes = metricsBuf[10].toInt()
        val chunks = metricsBuf[16].toInt()
        return totalTris >= 12 || chunks >= 1 || cachedPlanes >= 1
    }

    private fun autoSaveRoomCache(showBanner: Boolean) {
        if (relocPending || roomScanMode <= 0 || !hasScannedRoomData()) return
        roomStorage.saveAutoCache { res ->
            res.onSuccess { kb ->
                if (showBanner) {
                    showStereoBanner(
                        "\uD83D\uDD12  Đã khóa & tự lưu cache vào máy (${kb} KB) cho lần sau!",
                        "#00E676",
                        2200L,
                        strokeWidth = 4
                    )
                }
            }
        }
    }

    private fun maybeAutoLoadRoomCache() {
        if (autoCacheLoadedAttempted) return
        autoCacheLoadedAttempted = true
        if (!roomStorage.hasAutoCache()) return
        roomStorage.loadAutoCache { res ->
            res.onSuccess { (p, v, k) ->
                if (roomScanMode == 0) {
                    roomScanMode = 4
                    roomScanToggleBtn.text = roomScanLabel(roomScanMode)
                    NativeBridge.nativeSetRoomScanMode(roomScanMode)
                    updateArPlaneFindingMode()
                    startReticleHudLoop()
                }
                if (k > 0) {
                    relocPending = true
                    relocStatusLabel.text = "\u23F3 Tự khớp cache phòng cũ ($p mặt, $v ô)..."
                    relocBar.visibility = View.VISIBLE
                    showStereoBanner(
                        "\uD83D\uDCBE  Đã tự nạp phòng cũ từ máy — lia nhẹ để khớp hoặc bấm [📍 Khớp ngay] / [🧹 Xóa cache]",
                        "#FFB300",
                        2800L,
                        strokeWidth = 4
                    )
                } else {
                    NativeBridge.nativeForceRelocalizeNow()
                    relocPending = false
                    relocBar.visibility = View.GONE
                    scanTurbo = false
                    updateScanTurboButtonUi()
                    showStereoBanner("\uD83D\uDCBE  Đã tự nạp phòng cũ từ máy ([🔒 Khóa])", "#00E676", 2200L)
                }
            }
        }
    }

    private fun clearRoomCacheAndReset() {
        NativeBridge.nativeCancelRelocalize()
        NativeBridge.nativeClearSpatialMesh()
        relocPending = false
        relocBar.visibility = View.GONE
        lastDepthTimestampNs = 0L
        scanTurbo = true
        NativeBridge.nativeSetScanTurbo(true)
        updateScanTurboButtonUi()
        roomPanelLayout.visibility = View.GONE
        panelLayout.visibility = View.GONE
        roomStorage.clearAutoCache {
            showStereoBanner(
                "\uD83E\uDDF9  Đã xóa sạch cache cũ trên máy — bật [⚡ Quét mạnh] quét mới!",
                "#FFB300",
                2200L,
                strokeWidth = 4
            )
        }
    }

    private fun startAutoSaveLoop() {
        stopAutoSaveLoop()
        val task = object : Runnable {
            override fun run() {
                autoSaveRoomCache(showBanner = false)
                mainHandler.postDelayed(this, 20_000L)
            }
        }
        autoSaveRunnable = task
        mainHandler.postDelayed(task, 20_000L)
    }

    private fun stopAutoSaveLoop() {
        autoSaveRunnable?.let { mainHandler.removeCallbacks(it) }
        autoSaveRunnable = null
    }

    private fun updateScanTurboButtonUi() {
        scanTurboBtn.text = if (scanTurbo) "\u26A1 Quét mạnh" else "\uD83D\uDD12 Khóa"
        (scanTurboBtn.background as? GradientDrawable)?.setColor(
            if (scanTurbo) Color.argb(225, 217, 119, 6) else Color.argb(210, 14, 116, 144)
        )
    }

    private fun toggleScanTurbo(showPopup: Boolean) {
        scanTurbo = !scanTurbo
        if (scanTurbo && roomScanMode == 0) {
            roomScanMode = 4
            roomScanToggleBtn.text = roomScanLabel(roomScanMode)
            (roomScanToggleBtn.background as? GradientDrawable)?.setColor(Color.argb(215, 14, 138, 108))
            NativeBridge.nativeSetRoomScanMode(roomScanMode)
            updateArPlaneFindingMode()
            startReticleHudLoop()
        }
        NativeBridge.nativeSetScanTurbo(scanTurbo)
        updateScanTurboButtonUi()
        if (!scanTurbo) {
            // Locking room automatically persists current 3D map to on-phone cache for next launch!
            if (hasScannedRoomData()) {
                autoSaveRoomCache(showBanner = showPopup)
            } else if (showPopup) {
                showStereoBanner(
                    "\uD83D\uDD12  ĐÃ KHÓA LƯỚI (2Hz): Không quét lung tung — chỉ sửa khi vật thật đổi >1.5s!",
                    "#26C6DA",
                    2100L,
                    strokeWidth = 4
                )
            }
        } else if (showPopup) {
            showStereoBanner(
                "\u26A1  QUÉT MẠNH BẬT (9Hz): Lại gần (<1m) quét lưới 2.5cm siêu mịn & tự xóa lỗi lơ lửng!",
                "#FFB300",
                2100L,
                strokeWidth = 4
            )
        }
    }

    private fun updateTorchButtonUi() {
        torchBtn.text = if (torchEnabled) "\uD83D\uDD26 BẬT" else "\uD83D\uDD26 Đèn"
        (torchBtn.background as? GradientDrawable)?.setColor(
            if (torchEnabled) Color.argb(235, 234, 152, 12) else Color.argb(190, 24, 30, 42)
        )
    }

    private fun toggleTorch(showPopup: Boolean) {
        torchEnabled = !torchEnabled
        val ok = applyTorchMode()
        if (!ok && torchEnabled) {
            torchEnabled = false
        }
        updateTorchButtonUi()
        if (showPopup) {
            val banner = if (!ok) {
                "\u26A0\uFE0F  Không bật được đèn flash trên chế độ camera hiện tại"
            } else if (torchEnabled) {
                "\uD83D\uDD26  Đã BẬT đèn flash — camera nét hơn để quét chi tiết đồ vật!"
            } else {
                "\uD83D\uDD26  Đã TẮT đèn flash"
            }
            showStereoBanner(banner, if (torchEnabled) "#FFD54F" else "#90A4AE", 1700L)
        }
    }

    private fun applyTorchMode(): Boolean {
        synchronized(sessionLock) {
            val session = arSession
            if (session != null) {
                return try {
                    val cfg = session.config
                    cfg.flashMode = if (torchEnabled) Config.FlashMode.TORCH else Config.FlashMode.OFF
                    session.configure(cfg)
                    true
                } catch (t: Throwable) {
                    Log.w(TAG, "ARCore FlashMode warning: ${t.message}")
                    false
                }
            }
            val cs = captureSession
            val req = fallbackReqBuilder
            val handler = cameraHandler
            if (cs != null && req != null) {
                return try {
                    req.set(
                        CaptureRequest.FLASH_MODE,
                        if (torchEnabled) CaptureRequest.FLASH_MODE_TORCH else CaptureRequest.FLASH_MODE_OFF
                    )
                    cs.setRepeatingRequest(req.build(), null, handler)
                    true
                } catch (t: Throwable) {
                    Log.w(TAG, "Camera2 FlashMode warning: ${t.message}")
                    false
                }
            }
        }
        return true
    }

    private fun toggleStereoMode(showPopup: Boolean) {
        stereoMode = !stereoMode
        viewportDirty = true
        NativeBridge.nativeSetStereoMode(stereoMode)
        eyeModeBtn.text = if (stereoMode) "\uD83E\uDD7D 2 Mắt VR" else "\uD83D\uDC41 1 Mắt AR"
        applyStereoUiLayout()
        if (showPopup) {
            val banner = if (stereoMode) {
                "\uD83E\uDD7D  Chế độ 2 MẮT (VR Hộp Stereo)"
            } else {
                "\uD83D\uDC41  Chế độ 1 MẮT (Full màn AR chuẩn khoảng cách)"
            }
            showStereoBanner(banner, "#00E5FF", 1400L)
        }
    }

    private fun performUpdateCheck(manual: Boolean) {
        if (appUpdater.isDownloading) return
        if (manual) {
            checkUpdateBtn.text = "\u23F3"
            checkUpdateBtn.isEnabled = false
        }
        appUpdater.checkForUpdate { res ->
            checkUpdateBtn.text = "\uD83D\uDD04"
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
                installUpdateBtn.text = "\u2B07 $tag"
                installUpdateBtn.visibility = View.VISIBLE
                showStereoBanner(
                    "\u2B07  Có bản cập nhật mới ($tag)! Bấm [⬇ $tag] để cài",
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

    private fun cycleHandMode(showPopup: Boolean) {
        handMode = when (handMode) {
            0 -> 1
            1 -> 2
            else -> 0
        }
        handTracker.setMode(handMode)
        handModeBtn.text = handModeLabel(handMode)
        if (showPopup) {
            val banner = when (handMode) {
                2 -> "\uD83D\uDD90  AI Tay: NHANH (24Hz liên tục)"
                0 -> "\uD83D\uDD90  Đã TẮT AI Tay (Max FPS xuyên thấu)"
                else -> "\uD83D\uDD90  AI Tay: TỰ ĐỘNG (Tiết kiệm GPU)"
            }
            showStereoBanner(banner, "#69F0AE", 1500L)
        }
    }

    private fun cycleRoomScanMode(showPopup: Boolean) {
        roomScanMode = when (roomScanMode) {
            4 -> 3
            3 -> 0
            else -> 4
        }
        roomScanToggleBtn.text = roomScanLabel(roomScanMode)
        val btnBg = roomScanToggleBtn.background as? GradientDrawable
        btnBg?.setColor(
            if (roomScanMode == 4) Color.argb(215, 14, 138, 108) else Color.argb(175, 20, 24, 34)
        )
        NativeBridge.nativeSetRoomScanMode(roomScanMode)
        if (roomScanMode == 0) {
            relocPending = false
            relocBar.visibility = View.GONE
        }
        updateArPlaneFindingMode()

        if (roomScanMode > 0) {
            startReticleHudLoop()
            startAutoSaveLoop()
        } else {
            stopReticleHudLoop()
            stopAutoSaveLoop()
            leftReticleHud.visibility = View.GONE
            rightReticleHud.visibility = View.GONE
        }

        if (showPopup) {
            val banner = when (roomScanMode) {
                4 -> "\uD83D\uDD3A  Lưới thích ứng 2.5cm: Quét cả sàn + Gộp phẳng + Nhớ phòng 360°"
                3 -> "\uD83D\uDC41  Ẩn lưới (Giữ che khuất vật thật + Vật lý + Hồng tâm)"
                else -> "\u23F9  Đã TẮT quét không gian & hồng tâm"
            }
            showStereoBanner(banner, if (roomScanMode == 4) "#2EE6A8" else "#2EE6E6", 1700L)
        }
    }

    private fun startReticleHudLoop() {
        stopReticleHudLoop()
        if (roomScanMode <= 0) return
        val task = object : Runnable {
            override fun run() {
                if (roomScanMode <= 0) return
                updateReticleHudPills()
                mainHandler.postDelayed(this, 70L)
            }
        }
        reticleHudRunnable = task
        mainHandler.post(task)
    }

    private fun stopReticleHudLoop() {
        reticleHudRunnable?.let { mainHandler.removeCallbacks(it) }
        reticleHudRunnable = null
    }

    private fun surfaceShortName(kindId: Int): String = when (kindId) {
        2 -> "Tường"
        3 -> "Mặt phẳng"
        4 -> "Bề mặt"
        5 -> "Trần"
        6 -> "Vật thể"
        else -> "Đã quét"
    }

    private fun updateReticleHudPills() {
        if (roomScanMode <= 0) {
            leftReticleHud.visibility = View.GONE
            rightReticleHud.visibility = View.GONE
            return
        }
        NativeBridge.nativeGetPerfMetrics(metricsBuf)
        val hk = metricsBuf[13].toInt()
        val sk = metricsBuf[14].toInt()
        val dist = metricsBuf[15]
        var meshPrefix = ""
        if (roomScanMode == 4) {
            val detailTris = metricsBuf[24].toInt()
            val flatTris = metricsBuf[25].toInt()
            val diffChanged = metricsBuf[23].toInt()
            meshPrefix = if (scanTurbo) {
                if (blurSkipActive) {
                    "\u26A1 Giữ chậm (Chi tiết:${detailTris}\u25B3) \u00B7 "
                } else {
                    "\u26A1 Chi tiết:${detailTris}\u25B3 \u00B7 Phẳng:${flatTris}\u25B3 \u00B7 "
                }
            } else if (diffChanged > 0) {
                "\uD83D\uDD12 Đổi +${diffChanged} (Chi tiết:${detailTris}\u25B3) \u00B7 "
            } else {
                "\uD83D\uDD12 Khóa (Chi tiết:${detailTris}\u25B3 \u00B7 Phẳng:${flatTris}\u25B3) \u00B7 "
            }
        }
        if (relocPending) {
            val bestPct = metricsBuf[20].toInt()
            relocStatusLabel.text = "\u23F3 Tự khớp phòng (${bestPct}%) — lia quanh phòng hoặc bấm:"
            relocBar.visibility = View.VISIBLE
        }
        val (text, strokeCol) = if (hk > 0 && dist > 0.05f) {
            val sName = surfaceShortName(sk)
            String.format(Locale.US, "%s\uD83C\uDFAF %s \u00B7 %.2fm", meshPrefix, sName, dist) to Color.argb(215, 46, 230, 200)
        } else {
            "${meshPrefix}\uD83D\uDD0D Chưa quét bề mặt tại tâm" to Color.argb(145, 160, 180, 200)
        }

        leftReticleHud.text = text
        (leftReticleHud.background as? GradientDrawable)?.setStroke(2, strokeCol)
        leftReticleHud.visibility = View.VISIBLE

        if (stereoMode) {
            rightReticleHud.text = text
            (rightReticleHud.background as? GradientDrawable)?.setStroke(2, strokeCol)
            rightReticleHud.visibility = View.VISIBLE
        } else {
            rightReticleHud.visibility = View.GONE
        }
    }

    private fun toggleTelemetryHud() {
        telemetryVisible = !telemetryVisible
        leftTelemetryCard.visibility = if (telemetryVisible) View.VISIBLE else View.GONE
        rightTelemetryCard.visibility = if (telemetryVisible && stereoMode) View.VISIBLE else View.GONE
        telemetryToggleBtn.text = if (telemetryVisible) "\uD83D\uDCCA Thông số: BẬT" else "\uD83D\uDCCA Thông số: TẮT"
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
        val activePlanes = metricsBuf[8].toInt()
        val activeObjects = metricsBuf[9].toInt()
        val cachedPlanes = metricsBuf[10].toInt()
        val voxelCount = metricsBuf[11].toInt()
        val physBodies = metricsBuf[12].toInt()
        val hitKind = metricsBuf[13].toInt()
        val surfKind = metricsBuf[14].toInt()
        val hitDist = metricsBuf[15]
        val meshChunks = metricsBuf[16].toInt()
        val meshTris = metricsBuf[17].toInt()
        val kfCount = metricsBuf[18].toInt()
        val relocSt = metricsBuf[19].toInt()
        val relocPct = metricsBuf[20].toInt()

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
        val yuvUs = handTracker.lastConvertUs
        val shortSha = BuildConfig.GIT_COMMIT_SHA.take(7)

        val handAiText = if (handMode == 0) {
            "AI Tay: TẮT"
        } else {
            "AI Tay (${handTracker.delegateLabel}): $aiFps fps (${inferMs}ms | LUT ${yuvUs}µs)"
        }
        val viewLabel = if (stereoMode) "2 Mắt VR" else "1 Mắt AR"
        val reticleSummary = if (hitKind > 0 && hitDist > 0.05f) {
            String.format(Locale.US, "%s %.2fm", surfaceShortName(surfKind), hitDist)
        } else {
            "Chưa khóa"
        }

        val relocTag = when (relocSt) {
            1 -> "Khớp:${relocPct}%"
            2 -> "Tự khớp OK"
            3 -> "Khớp tay OK"
            else -> "Mốc:${kfCount}/16"
        }

        val depthModeAct = metricsBuf[21].toInt()
        val diffChanged = metricsBuf[23].toInt()
        val detailTris = metricsBuf[24].toInt()
        val flatTris = metricsBuf[25].toInt()
        val depthSrc = when {
            depthModeAct == 2 -> "DenseDepth 2.5cm"
            depthModeAct == 1 -> "RawDepth 2.5cm"
            depthSupportLevel == 2 -> "DenseDepth(chờ)"
            depthSupportLevel == 1 -> "RawDepth(chờ)"
            else -> "SLAM"
        }
        val scanTag = if (scanTurbo) {
            if (blurSkipActive) "\u26A1Mạnh(Chống mờ)" else "\u26A1Mạnh(9Hz)"
        } else if (diffChanged > 0) {
            "\uD83D\uDD12Khóa(Đổi +${diffChanged})"
        } else {
            "\uD83D\uDD12Khóa(0 đổi)"
        }
        val occTag = if (meshOcclusionEnabled) "Che:BẬT" else "Che:TẮT"
        val torchTag = if (torchEnabled) "Đèn:BẬT" else "Đèn:TẮT"

        val summary = StringBuilder()
            .append("v${BuildConfig.VERSION_NAME} ($shortSha) | $viewLabel | ${batPct}% ${String.format(Locale.US, "%.1f", tempC)}°C | ${totalRamMb}MB\n")
            .append(String.format(Locale.US, "GPU: %d fps (%.1fms) | %s | %s | %s\n", renderFps, frameMs, depthSrc, occTag, torchTag))
            .append("Track: $trackMode | $scanTag | Tâm: $reticleSummary | Vật lý: $physBodies\n")
            .append("Chi tiết: ${detailTris}\u25B3 ($meshChunks chunk) | Phẳng gộp: ${flatTris}\u25B3 | $relocTag\n")
            .append("Tay ($activeHands): $gestureName | $handAiText")
            .toString()

        leftTelemetryCard.text = summary
        if (stereoMode) {
            rightTelemetryCard.text = summary
        }
    }

    private fun showStereoGesturePopup(code: Int) {
        val (label, accentHex, durationMs) = when (code) {
            1 -> Triple("\u270A  Phát hiện: NẮM ĐẤM", "#FF7043", 1800L)
            2 -> Triple("\uD83E\uDD0F  Phát hiện: CHỤM NGÓN CÁI", "#00E5FF", 1800L)
            3 -> Triple("\uD83D\uDC49  CÂY SÚNG: Bắn bóng vật lý!", "#FFD54F", 1800L)
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
            if (pill === rightGesturePopup && !stereoMode) {
                pill.visibility = View.GONE
                continue
            }
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

    private fun addRoomManagerOverlay(root: FrameLayout) {
        // Floating bottom-left relocalization control bar (visible only while aligning a loaded room)
        val relocBg = GradientDrawable().apply {
            cornerRadius = 22f
            setColor(Color.argb(220, 20, 26, 40))
            setStroke(2, Color.parseColor("#FFB300"))
        }
        relocBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            background = relocBg
            setPadding(dp(8f), dp(5f), dp(8f), dp(5f))
            visibility = View.GONE
        }
        relocStatusLabel = TextView(this).apply {
            setTextColor(Color.parseColor("#FFE082"))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 11.5f)
            typeface = Typeface.DEFAULT_BOLD
            text = "\u23F3 Tự khớp phòng..."
        }
        val lockNowBtn = makeQuickButton("\uD83D\uDCCD Khớp ngay", fillColor = Color.argb(230, 217, 119, 6)) {
            val ok = NativeBridge.nativeForceRelocalizeNow()
            relocPending = false
            relocBar.visibility = View.GONE
            if (ok) {
                scanTurbo = false
                updateScanTurboButtonUi()
                showStereoBanner("\uD83D\uDCCD  Đã khớp phòng cũ — đang ở [🔒 Khóa]!", "#00E676", 2400L)
            }
        }
        val clearCacheRelocBtn = makeQuickButton("\uD83E\uDDF9 Xóa cache", fillColor = Color.argb(215, 165, 65, 24)) {
            clearRoomCacheAndReset()
        }
        val cancelRelocBtn = makeQuickButton("\u2716", fillColor = Color.argb(190, 90, 35, 35)) {
            NativeBridge.nativeCancelRelocalize()
            relocPending = false
            relocBar.visibility = View.GONE
            showStereoBanner("Bỏ qua khớp phòng cũ (vẫn giữ cache trong máy)", "#FFAB40", 1500L)
        }
        val rItemLp = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.WRAP_CONTENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply { marginStart = dp(5f) }
        relocBar.addView(relocStatusLabel)
        relocBar.addView(lockNowBtn, rItemLp)
        relocBar.addView(clearCacheRelocBtn, rItemLp)
        relocBar.addView(cancelRelocBtn, rItemLp)

        root.addView(
            relocBar,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.WRAP_CONTENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.BOTTOM or Gravity.START
            ).apply {
                bottomMargin = dp(10f)
                marginStart = dp(10f)
            }
        )

        // Centered Room Manager Modal Panel
        val panelBg = GradientDrawable().apply {
            cornerRadius = 24f
            setColor(Color.argb(238, 14, 18, 28))
            setStroke(2, Color.argb(180, 46, 230, 200))
        }
        roomPanelLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            visibility = View.GONE
            background = panelBg
            setPadding(dp(14f), dp(12f), dp(14f), dp(12f))
        }

        val title = TextView(this).apply {
            setTextColor(Color.WHITE)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 13.5f)
            typeface = Typeface.DEFAULT_BOLD
            text = "\uD83D\uDCBE Bản đồ Phòng 3D (.arroom & .obj)"
        }
        roomStatusText = TextView(this).apply {
            setTextColor(Color.parseColor("#B2EBF2"))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 11.5f)
            setPadding(0, dp(4f), 0, dp(8f))
            text = "Đang tải..."
        }

        val actionRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val saveNewBtn = makeQuickButton("\uD83D\uDCBE Lưu phòng mới", fillColor = Color.argb(225, 14, 138, 108)) {
            val autoName = roomStorage.nextAutoRoomName()
            roomStatusText.text = "Đang nén & lưu $autoName.arroom..."
            roomStorage.saveCurrentRoom(autoName) { res ->
                res.onSuccess { info ->
                    showStereoBanner(
                        "\uD83D\uDCBE  Đã lưu ${info.name}.arroom (${info.sizeKb} KB)",
                        "#00E676",
                        2000L
                    )
                    refreshSavedRoomsList()
                }.onFailure { err ->
                    showStereoBanner("\u26A0\uFE0F  ${err.message}", "#FF7043", 2200L)
                    refreshSavedRoomsList()
                }
            }
        }
        val exportObjBtn = makeQuickButton("\uD83D\uDCE4 Xuất .OBJ", fillColor = Color.argb(225, 42, 92, 165)) {
            val baseName = "phong_${System.currentTimeMillis() / 1000L}"
            roomStatusText.text = "Đang xuất $baseName.obj..."
            roomStorage.exportCurrentRoomObj(baseName) { res ->
                res.onSuccess { file ->
                    val kb = ((file.length() + 1023L) / 1024L).coerceAtLeast(1L)
                    showStereoBanner("\uD83D\uDCE4  Đã xuất ${file.name} (${kb} KB)", "#40C4FF", 2400L)
                    refreshSavedRoomsList()
                }.onFailure { err ->
                    showStereoBanner("\u26A0\uFE0F  ${err.message}", "#FF7043", 2200L)
                    refreshSavedRoomsList()
                }
            }
        }
        val clearMeshBtn = makeQuickButton("\uD83E\uDDF9 Xóa cache cũ", fillColor = Color.argb(225, 178, 56, 32)) {
            clearRoomCacheAndReset()
        }
        val closeBtn = makeQuickButton("\u2716 Đóng", fillColor = Color.argb(200, 80, 35, 40)) {
            roomPanelLayout.visibility = View.GONE
        }
        val aLp = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.WRAP_CONTENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply { marginEnd = dp(6f) }
        actionRow.addView(saveNewBtn, aLp)
        actionRow.addView(exportObjBtn, aLp)
        actionRow.addView(clearMeshBtn, aLp)
        actionRow.addView(closeBtn)

        roomListContainer = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, dp(6f), 0, dp(2f))
        }
        val scroll = ScrollView(this).apply {
            addView(roomListContainer)
        }
        val scrollLp = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            dp(150f)
        ).apply { topMargin = dp(6f) }

        val pathHint = TextView(this).apply {
            setTextColor(Color.parseColor("#90A4AE"))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 8.5f)
            setPadding(0, dp(4f), 0, 0)
            text = "Tự lưu cache vào bộ nhớ máy khi bấm [🔒 Khóa] / thoát app. Xuất .OBJ tại: ${roomStorage.storageFolderPath()}"
        }

        roomPanelLayout.addView(title)
        roomPanelLayout.addView(roomStatusText)
        roomPanelLayout.addView(actionRow)
        roomPanelLayout.addView(scroll, scrollLp)
        roomPanelLayout.addView(pathHint)

        root.addView(
            roomPanelLayout,
            FrameLayout.LayoutParams(
                dp(520f),
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.CENTER
            )
        )
    }

    private fun toggleRoomManagerPanel() {
        panelLayout.visibility = View.GONE
        if (roomPanelLayout.visibility == View.VISIBLE) {
            roomPanelLayout.visibility = View.GONE
        } else {
            roomPanelLayout.visibility = View.VISIBLE
            refreshSavedRoomsList()
        }
    }

    private fun refreshSavedRoomsList() {
        NativeBridge.nativeGetPerfMetrics(metricsBuf)
        val planes = metricsBuf[10].toInt()
        val chunks = metricsBuf[16].toInt()
        val kfs = metricsBuf[18].toInt()
        roomStatusText.text = "Hiện tại: $chunks chunk 3D | $kfs/16 mốc ảnh định vị\n${roomStorage.autoCacheSummaryText()}"

        roomStorage.listRooms { rooms ->
            roomListContainer.removeAllViews()
            if (rooms.isEmpty()) {
                val empty = TextView(this).apply {
                    setTextColor(Color.LTGRAY)
                    setTextSize(TypedValue.COMPLEX_UNIT_SP, 10f)
                    setPadding(dp(4f), dp(12f), dp(4f), dp(12f))
                    text = "Chưa có phòng nào được lưu. Quét quanh phòng rồi bấm [💾 Lưu phòng mới]."
                }
                roomListContainer.addView(empty)
                return@listRooms
            }

            for (info in rooms) {
                val row = LinearLayout(this).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    setPadding(dp(6f), dp(4f), dp(6f), dp(4f))
                    setBackgroundColor(Color.argb(95, 255, 255, 255))
                }
                val label = TextView(this).apply {
                    setTextColor(Color.WHITE)
                    setTextSize(TypedValue.COMPLEX_UNIT_SP, 10f)
                    text = "${info.name}  (${info.sizeKb} KB · ${info.dateLabel})"
                }
                val labelLp = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)

                val loadBtn = makeQuickButton("\uD83D\uDCC2 Nạp", fillColor = Color.argb(215, 20, 120, 150)) {
                    roomStorage.loadRoom(info.file) { res ->
                        res.onSuccess { (p, v, k) ->
                            roomPanelLayout.visibility = View.GONE
                            if (roomScanMode == 0) {
                                roomScanMode = 4
                                roomScanToggleBtn.text = roomScanLabel(roomScanMode)
                                NativeBridge.nativeSetRoomScanMode(roomScanMode)
                                updateArPlaneFindingMode()
                                startReticleHudLoop()
                            }
                            if (k > 0) {
                                relocPending = true
                                relocStatusLabel.text = "\u23F3 Đang tự khớp ${info.name} ($p mặt, $v ô)..."
                                relocBar.visibility = View.VISIBLE
                                showStereoBanner(
                                    "\uD83D\uDCC2  Đã nạp ${info.name}: lia quanh phòng để tự khớp hoặc bấm [📍 Khớp ngay]",
                                    "#FFB300",
                                    2600L
                                )
                            } else {
                                NativeBridge.nativeForceRelocalizeNow()
                                relocPending = false
                                relocBar.visibility = View.GONE
                                scanTurbo = false
                                updateScanTurboButtonUi()
                                showStereoBanner("\uD83D\uDCCD  Đã nạp & khớp ${info.name} — đang ở [🔒 Khóa]", "#00E676", 2200L)
                            }
                        }.onFailure { err ->
                            showStereoBanner("\u26A0\uFE0F  ${err.message}", "#FF5252", 2200L)
                        }
                    }
                }
                val overwriteBtn = makeQuickButton("\uD83D\uDCBE Đè", fillColor = Color.argb(200, 18, 110, 82)) {
                    roomStorage.saveCurrentRoom(info.name) { res ->
                        res.onSuccess { updated ->
                            showStereoBanner("\uD83D\uDCBE  Đã cập nhật ${updated.name}.arroom (${updated.sizeKb} KB)", "#00E676", 1800L)
                            refreshSavedRoomsList()
                        }.onFailure { err ->
                            showStereoBanner("\u26A0\uFE0F  ${err.message}", "#FF7043", 2000L)
                        }
                    }
                }
                val delBtn = makeQuickButton("\uD83D\uDDD1", fillColor = Color.argb(180, 110, 35, 35)) {
                    roomStorage.deleteRoom(info.file) { refreshSavedRoomsList() }
                }
                val bLp = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { marginStart = dp(4f) }

                row.addView(label, labelLp)
                row.addView(loadBtn, bLp)
                row.addView(overwriteBtn, bLp)
                row.addView(delBtn, bLp)

                val rowLp = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { bottomMargin = dp(4f) }
                roomListContainer.addView(row, rowLp)
            }
        }
    }

    private fun addCalibrationOverlay(root: FrameLayout) {
        val drawerBg = GradientDrawable().apply {
            cornerRadius = 26f
            setColor(Color.argb(242, 16, 20, 30))
            setStroke(2, Color.argb(175, 0, 229, 255))
        }
        panelLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            visibility = View.GONE
            background = drawerBg
            setPadding(dp(16f), dp(14f), dp(16f), dp(14f))
        }

        val headerRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val titleTxt = TextView(this).apply {
            setTextColor(Color.WHITE)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 14.5f)
            typeface = Typeface.DEFAULT_BOLD
            text = "\u2699 Cài đặt & Tùy chọn nâng cao"
        }
        val closeSettingsBtn = makeQuickButton("\u2716 Đóng", fillColor = Color.argb(215, 110, 36, 44)) {
            panelLayout.visibility = View.GONE
        }
        headerRow.addView(titleTxt, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        headerRow.addView(closeSettingsBtn)

        val contentBox = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, dp(8f), 0, dp(4f))
        }

        // Row 1: 1/2 Eye Mode + Hand Tracking AI (BIG touch targets!)
        val row1 = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            weightSum = 2f
        }
        eyeModeBtn = makeQuickButton(if (stereoMode) "\uD83E\uDD7D 2 Mắt VR" else "\uD83D\uDC41 1 Mắt AR", Color.argb(220, 26, 72, 118)) {
            toggleStereoMode(showPopup = true)
        }.apply {
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12.5f)
            minHeight = dp(44f)
            minimumHeight = dp(44f)
        }
        handModeBtn = makeQuickButton(handModeLabel(handMode), Color.argb(220, 32, 82, 74)) {
            cycleHandMode(showPopup = true)
        }.apply {
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12.5f)
            minHeight = dp(44f)
            minimumHeight = dp(44f)
        }
        val halfLpL = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
            marginEnd = dp(6f)
        }
        val halfLpR = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
            marginStart = dp(6f)
        }
        row1.addView(eyeModeBtn, halfLpL)
        row1.addView(handModeBtn, halfLpR)

        // Row 2: Finger Draw + Telemetry HUD
        val row2 = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            weightSum = 2f
        }
        drawToggleBtn = makeQuickButton("\u270F Vẽ: TẮT", Color.argb(220, 62, 52, 108)) {
            cycleDrawMode(showPopup = true)
        }.apply {
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12.5f)
            minHeight = dp(44f)
            minimumHeight = dp(44f)
        }
        telemetryToggleBtn = makeQuickButton(
            if (telemetryVisible) "\uD83D\uDCCA Thông số: BẬT" else "\uD83D\uDCCA Thông số: TẮT",
            Color.argb(220, 44, 62, 86)
        ) {
            toggleTelemetryHud()
        }.apply {
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12.5f)
            minHeight = dp(44f)
            minimumHeight = dp(44f)
        }
        row2.addView(drawToggleBtn, halfLpL)
        row2.addView(telemetryToggleBtn, halfLpR)

        val rowLp = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply { topMargin = dp(8f) }

        val occToggleBtn = makeQuickButton(
            if (meshOcclusionEnabled) "\uD83D\uDEE1 Che khuất vật thật (Z-Buffer): ĐANG BẬT" else "\uD83D\uDEE1 Che khuất vật thật: ĐANG TẮT",
            Color.argb(220, 16, 108, 124)
        ) {}.apply {
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12.5f)
            minHeight = dp(44f)
            minimumHeight = dp(44f)
            setOnClickListener {
                meshOcclusionEnabled = !meshOcclusionEnabled
                NativeBridge.nativeSetMeshOcclusion(meshOcclusionEnabled)
                text = if (meshOcclusionEnabled) "\uD83D\uDEE1 Che khuất vật thật (Z-Buffer): ĐANG BẬT" else "\uD83D\uDEE1 Che khuất vật thật: ĐANG TẮT"
                showStereoBanner(
                    if (meshOcclusionEnabled) "\uD83D\uDEE1  Đã BẬT che khuất theo lưới 3D thật" else "\uD83D\uDEE1  Đã TẮT che khuất (chế độ xuyên thấu lưới)",
                    "#00E5FF",
                    1500L
                )
            }
        }

        val clearCacheSettingsBtn = makeQuickButton(
            "\uD83E\uDDF9 Xóa cache phòng 3D cũ trên máy & quét mới",
            Color.argb(225, 178, 56, 32)
        ) {
            clearRoomCacheAndReset()
        }.apply {
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12.5f)
            minHeight = dp(44f)
            minimumHeight = dp(44f)
        }

        val fovLabel = TextView(this).apply {
            setTextColor(Color.WHITE)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
            setPadding(0, dp(10f), 0, 0)
            text = "Tỷ lệ khung hình (100% = Full màn): $fovPct%"
        }
        val fovBar = SeekBar(this).apply {
            max = 50
            progress = fovPct - 55
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                    fovPct = 55 + p
                    fovLabel.text = "Tỷ lệ khung hình (100% = Full màn): $fovPct%"
                    saveCalibration()
                }
                override fun onStartTrackingTouch(sb: SeekBar?) {}
                override fun onStopTrackingTouch(sb: SeekBar?) {}
            })
        }

        val ipdLabel = TextView(this).apply {
            setTextColor(Color.WHITE)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
            setPadding(0, dp(8f), 0, 0)
            text = "Khoảng cách 2 mắt VR (IPD): $ipdMm mm"
        }
        val ipdBar = SeekBar(this).apply {
            max = 16
            progress = ipdMm - 56
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                    ipdMm = 56 + p
                    ipdLabel.text = "Khoảng cách 2 mắt VR (IPD): $ipdMm mm"
                    saveCalibration()
                }
                override fun onStartTrackingTouch(sb: SeekBar?) {}
                override fun onStopTrackingTouch(sb: SeekBar?) {}
            })
        }

        val tipText = TextView(this).apply {
            setTextColor(Color.parseColor("#B0BEC5"))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
            setPadding(0, dp(8f), 0, 0)
            text = "Lưới thích ứng 2.5cm: Đưa máy lại gần đồ vật (<1m) để quét mịn; sàn, tường & mặt bàn tự gộp phẳng; nhớ trọn phòng 360° & tự lưu vào máy.\nMẹo VR: Volume+ = Đặt lại | Volume- = Đổi chế độ vẽ tay"
        }

        contentBox.addView(row1)
        contentBox.addView(row2, rowLp)
        contentBox.addView(occToggleBtn, rowLp)
        contentBox.addView(clearCacheSettingsBtn, rowLp)
        contentBox.addView(fovLabel)
        contentBox.addView(fovBar)
        contentBox.addView(ipdLabel)
        contentBox.addView(ipdBar)
        contentBox.addView(tipText)

        val settingsScroll = ScrollView(this).apply {
            addView(contentBox)
        }
        panelLayout.addView(headerRow)
        panelLayout.addView(
            settingsScroll,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                dp(275f)
            )
        )

        val panelParams = FrameLayout.LayoutParams(
            dp(500f),
            FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.CENTER
        )
        root.addView(panelLayout, panelParams)
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
        if (roomScanMode > 0) {
            startReticleHudLoop()
            startAutoSaveLoop()
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
        hidePopupRunnable?.let { mainHandler.removeCallbacks(it) }
        stopReticleHudLoop()
        stopAutoSaveLoop()
        stopTelemetryLoop()
        autoSaveRoomCache(showBanner = false)
        sensorManager.unregisterListener(this)
        if (torchEnabled) {
            torchEnabled = false
            applyTorchMode()
            updateTorchButtonUi()
        }
        stopTrackingAndCamera()
        glView.onPause()
        super.onPause()
    }

    override fun onDestroy() {
        handTracker.close()
        roomStorage.close()
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
                    val autoDepthOk = try {
                        session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)
                    } catch (_: Throwable) {
                        false
                    }
                    val rawDepthOk = try {
                        session.isDepthModeSupported(Config.DepthMode.RAW_DEPTH_ONLY)
                    } catch (_: Throwable) {
                        false
                    }
                    depthSupportLevel = when {
                        autoDepthOk -> 2
                        rawDepthOk -> 1
                        else -> 0
                    }
                    val config = Config(session).apply {
                        focusMode = Config.FocusMode.AUTO
                        planeFindingMode = Config.PlaneFindingMode.DISABLED
                        lightEstimationMode = Config.LightEstimationMode.DISABLED
                        instantPlacementMode = Config.InstantPlacementMode.DISABLED
                        depthMode = when {
                            roomScanMode <= 0 -> Config.DepthMode.DISABLED
                            depthSupportLevel == 2 -> Config.DepthMode.AUTOMATIC
                            depthSupportLevel == 1 -> Config.DepthMode.RAW_DEPTH_ONLY
                            else -> Config.DepthMode.DISABLED
                        }
                        flashMode = if (torchEnabled) Config.FlashMode.TORCH else Config.FlashMode.OFF
                        updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
                    }
                    session.configure(config)
                    arSession = session
                    arNeedsTextureBind = true
                    isUsingArCore = true
                    Log.i(TAG, "ARCore 6DoF initialized (planes=${config.planeFindingMode}, depth=${config.depthMode})")
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

    private fun updateArPlaneFindingMode() {
        synchronized(sessionLock) {
            val session = arSession ?: return
            try {
                val cfg = session.config
                val target = Config.PlaneFindingMode.DISABLED
                val targetDepth = when {
                    roomScanMode <= 0 -> Config.DepthMode.DISABLED
                    depthSupportLevel == 2 -> Config.DepthMode.AUTOMATIC
                    depthSupportLevel == 1 -> Config.DepthMode.RAW_DEPTH_ONLY
                    else -> Config.DepthMode.DISABLED
                }
                val targetFlash = if (torchEnabled) Config.FlashMode.TORCH else Config.FlashMode.OFF
                if (cfg.planeFindingMode != target || cfg.depthMode != targetDepth || cfg.flashMode != targetFlash) {
                    cfg.planeFindingMode = target
                    cfg.depthMode = targetDepth
                    cfg.flashMode = targetFlash
                    session.configure(cfg)
                }
            } catch (t: Throwable) {
                Log.w(TAG, "PlaneFindingMode update warning: ${t.message}")
            }
        }
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
                val wantHand = handMode > 0 && !handTracker.isBusy.get()
                var ingestedSeq = 0L
                try {
                    if (wantHand) {
                        val rot = displayRotation
                        val rotDeg = if (rot == Surface.ROTATION_270) 180 else 0
                        val basis = if (rot == Surface.ROTATION_270) fallbackBasis270 else fallbackBasis90
                        ingestedSeq = ingestYuvToNative(img, rotDeg, basis)
                    }
                } finally {
                    img.close()
                }
                if (ingestedSeq > 0L && wantHand) {
                    handTracker.notifyFrameAvailable()
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
                set(
                    CaptureRequest.FLASH_MODE,
                    if (torchEnabled) CaptureRequest.FLASH_MODE_TORCH else CaptureRequest.FLASH_MODE_OFF
                )
            }
            synchronized(sessionLock) {
                fallbackReqBuilder = req
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
        fallbackReqBuilder = null

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

    private fun ingestYuvToNative(img: Image, rotDeg: Int, basis: FloatArray): Long {
        val planes = img.planes
        if (planes.size < 3) return 0L
        val yBuf = planes[0].buffer
        val uBuf = planes[1].buffer
        val vBuf = planes[2].buffer
        yBuf.position(0)
        uBuf.position(0)
        vBuf.position(0)
        return NativeBridge.nativeIngestCameraImage(
            yBuf,
            uBuf,
            vBuf,
            yBuf.remaining(),
            uBuf.remaining(),
            vBuf.remaining(),
            img.width,
            img.height,
            planes[0].rowStride,
            planes[0].pixelStride,
            planes[1].rowStride,
            planes[1].pixelStride,
            rotDeg,
            basis
        )
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
                    val geomW = if (stereoMode) (viewWidth / 2).coerceAtLeast(1) else viewWidth.coerceAtLeast(1)
                    if (viewportDirty) {
                        session.setDisplayGeometry(displayRotation, geomW, viewHeight)
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
                                val sensorPoseNow = cam.pose
                                if (valid && hasPrevArPose && ts > prevArTsNs) {
                                    val dtSec = ((ts - prevArTsNs) * 1e-9).toFloat().coerceIn(0.01f, 0.25f)
                                    val dx = sensorPoseNow.tx() - prevArPosX
                                    val dy = sensorPoseNow.ty() - prevArPosY
                                    val dz = sensorPoseNow.tz() - prevArPosZ
                                    camLinSpeedMS = kotlin.math.sqrt(dx * dx + dy * dy + dz * dz) / dtSec
                                    val qDot = abs(
                                        sensorPoseNow.qx() * prevArQx +
                                            sensorPoseNow.qy() * prevArQy +
                                            sensorPoseNow.qz() * prevArQz +
                                            sensorPoseNow.qw() * prevArQw
                                    ).coerceIn(0f, 1f)
                                    camAngSpeedRadS = (2f * kotlin.math.acos(qDot)) / dtSec
                                    val turnLimitAng = if (scanTurbo) 0.90f else 0.75f
                                    val turnLimitLin = if (scanTurbo) 0.90f else 0.75f
                                    if (camAngSpeedRadS > turnLimitAng || camLinSpeedMS > turnLimitLin) {
                                        // 240ms post-turn settle cooldown: waits for ARCore Depth temporal filter
                                        // to converge after turning so laggy depth NEVER scrambles scanned rooms!
                                        blurSettleUntilNs = nowNs + 240_000_000L
                                    }
                                }
                                if (valid) {
                                    prevArPosX = sensorPoseNow.tx()
                                    prevArPosY = sensorPoseNow.ty()
                                    prevArPosZ = sensorPoseNow.tz()
                                    prevArQx = sensorPoseNow.qx()
                                    prevArQy = sensorPoseNow.qy()
                                    prevArQz = sensorPoseNow.qz()
                                    prevArQw = sensorPoseNow.qw()
                                    prevArTsNs = ts
                                    hasPrevArPose = true
                                } else {
                                    hasPrevArPose = false
                                    camAngSpeedRadS = 99f
                                    camLinSpeedMS = 99f
                                }

                                cam.getProjectionMatrix(arProjMat, 0, 0.02f, 50f)
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

                                // Fallback SLAM 3D PointCloud (only active in Turbo scan on non-Depth devices / before Depth warms up)
                                val pcIntervalNs = if (depthSupportLevel > 0) 500_000_000L else 200_000_000L
                                if (roomScanMode > 0 && valid && scanTurbo && nowNs - lastPointCloudNs >= pcIntervalNs) {
                                    lastPointCloudNs = nowNs
                                    try {
                                        val pc = frame.acquirePointCloud()
                                        try {
                                            val pts = pc.points
                                            pts.position(0)
                                            val totalPts = (pts.remaining() / 4).coerceAtMost(512)
                                            if (totalPts > 0) {
                                                pts.get(pointCloudBuf, 0, totalPts * 4)
                                                NativeBridge.nativeOnPointCloud(totalPts, pointCloudBuf)
                                            }
                                        } finally {
                                            pc.close()
                                        }
                                    } catch (_: Throwable) {
                                    }
                                }

                                // Primary Adaptive 2.5cm TSDF Depth Pipeline (Zero-Copy Direct ByteBuffer -> Rust):
                                // - Turbo Scan ON:  ~9Hz (110ms) with 240ms post-turn settle guard & protected room memory
                                // - Locked Mode:    2.0Hz (500ms), TRUE LOCK — ignores wobble & unscanned space, 3-tick confirm!
                                val depthIntervalNs = if (scanTurbo) 110_000_000L else 500_000_000L
                                if (roomScanMode > 0 && valid && depthSupportLevel > 0 && nowNs - lastRawDepthNs >= depthIntervalNs) {
                                    val maxAngVel = if (scanTurbo) 0.90f else 0.75f
                                    val maxLinVel = if (scanTurbo) 0.90f else 0.75f
                                    if (camAngSpeedRadS > maxAngVel || camLinSpeedMS > maxLinVel || nowNs < blurSettleUntilNs) {
                                        // Motion-blur & 240ms post-turn settle protection: prevents scrambled turn-back layers!
                                        blurSkipActive = true
                                    } else {
                                        blurSkipActive = false
                                        lastRawDepthNs = nowNs
                                        try {
                                            val isRawOnly = depthSupportLevel == 1
                                            val depthImg = if (isRawOnly) {
                                                frame.acquireRawDepthImage16Bits()
                                            } else {
                                                frame.acquireDepthImage16Bits()
                                            }
                                            var confImg: Image? = null
                                            try {
                                                val dTs = depthImg.timestamp
                                                if (dTs != lastDepthTimestampNs) {
                                                    lastDepthTimestampNs = dTs
                                                    confImg = try {
                                                        frame.acquireRawDepthConfidenceImage()
                                                    } catch (_: Throwable) {
                                                        null
                                                    }
                                                    if (!(isRawOnly && confImg == null)) {
                                                        val intrinsics = cam.imageIntrinsics
                                                        intrinsics.getFocalLength(depthFocalBuf, 0)
                                                        intrinsics.getPrincipalPoint(depthPpBuf, 0)
                                                        intrinsics.getImageDimensions(depthDimsBuf, 0)

                                                        val dPlane = depthImg.planes[0]
                                                        val dBuf = dPlane.buffer
                                                        dBuf.position(0)
                                                        val dLen = dBuf.remaining()

                                                        val hasConf = confImg != null
                                                        val cPlane = confImg?.planes?.get(0)
                                                        val cBuf = cPlane?.buffer ?: dBuf
                                                        if (hasConf) cBuf.position(0)
                                                        val cLen = if (hasConf) cBuf.remaining() else 0

                                                        NativeBridge.nativeIngestDepthMap(
                                                            dBuf,
                                                            dLen,
                                                            depthImg.width,
                                                            depthImg.height,
                                                            dPlane.rowStride,
                                                            dPlane.pixelStride,
                                                            hasConf,
                                                            cBuf,
                                                            cLen,
                                                            confImg?.width ?: 0,
                                                            confImg?.height ?: 0,
                                                            cPlane?.rowStride ?: 1,
                                                            cPlane?.pixelStride ?: 1,
                                                            isRawOnly,
                                                            depthFocalBuf[0],
                                                            depthFocalBuf[1],
                                                            depthPpBuf[0],
                                                            depthPpBuf[1],
                                                            depthDimsBuf[0],
                                                            depthDimsBuf[1],
                                                            sensorPoseNow.tx(),
                                                            sensorPoseNow.ty(),
                                                            sensorPoseNow.tz(),
                                                            sensorPoseNow.qx(),
                                                            sensorPoseNow.qy(),
                                                            sensorPoseNow.qz(),
                                                            sensorPoseNow.qw()
                                                        )
                                                    }
                                                }
                                            } finally {
                                                try {
                                                    confImg?.close()
                                                } catch (_: Throwable) {}
                                                depthImg.close()
                                            }
                                        } catch (_: Throwable) {
                                        }
                                    }
                                }

                                // Ingest single 480p YUV frame into Rust 4-slot RAM ring (<0.08ms), then wake hand worker / relocalizer
                                val wantHand = handMode > 0 && !handTracker.isBusy.get()
                                val relocIntervalNs = if (relocPending) 250_000_000L else 1_000_000_000L
                                val wantReloc = roomScanMode > 0 && valid && (nowNs - lastRelocIngestNs >= relocIntervalNs)
                                if (wantHand || wantReloc) {
                                    var ingestedSeq = 0L
                                    try {
                                        frame.transformCoordinates2d(
                                            Coordinates2d.IMAGE_NORMALIZED,
                                            imgBasis,
                                            Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES,
                                            ndcBasis
                                        )
                                        val rotDeg = if (displayRotation == Surface.ROTATION_270) 180 else 0
                                        val cpuImg = frame.acquireCameraImage()
                                        try {
                                            ingestedSeq = ingestYuvToNative(cpuImg, rotDeg, ndcBasis)
                                        } finally {
                                            cpuImg.close()
                                        }
                                    } catch (_: Throwable) {
                                    }
                                    if (ingestedSeq > 0L) {
                                        if (wantHand) handTracker.notifyFrameAvailable()
                                        if (wantReloc) {
                                            lastRelocIngestNs = nowNs
                                            val rStatus = NativeBridge.nativeTickRelocalizer()
                                            if (relocPending && rStatus == 2) {
                                                relocPending = false
                                                mainHandler.post {
                                                    relocBar.visibility = View.GONE
                                                    scanTurbo = false
                                                    updateScanTurboButtonUi()
                                                    NativeBridge.nativeGetPerfMetrics(metricsBuf)
                                                    val pct = metricsBuf[20].toInt()
                                                    showStereoBanner(
                                                        "\u2705  Đã tự khớp phòng (${pct}%) — chuyển sang [🔒 Khóa]!",
                                                        "#00E676",
                                                        2400L,
                                                        strokeWidth = 4
                                                    )
                                                }
                                            }
                                        }
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
