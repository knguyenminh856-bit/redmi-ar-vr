package com.minhkhoi.arvr

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.SurfaceTexture
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.opengl.GLSurfaceView
import android.os.Build
import android.os.Bundle
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

    // Camera2 fallback (when Google Play Services for AR is unavailable)
    private var cameraThread: HandlerThread? = null
    private var cameraHandler: Handler? = null
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var fallbackSurfaceTexture: SurfaceTexture? = null
    private var fallbackPreviewSurface: Surface? = null
    private val fallbackFrameAvailable = AtomicBoolean(false)
    private var fallbackCameraActive = false
    private var fallbackTanHalfH = 0.67f
    private var fallbackTanHalfV = 0.50f
    private val fallbackTexMtx = FloatArray(16)

    // GL state synced with render thread
    @Volatile
    private var cameraTexId: Int = 0
    private var viewportDirty = false
    private var viewWidth = 2400
    private var viewHeight = 1080
    @Volatile
    private var displayRotation = Surface.ROTATION_90

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

    // Calibration UI state
    private var fovPct = 78
    private var ipdMm = 64
    private lateinit var panelLayout: LinearLayout
    private var recenterRunnable: Runnable? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        NativeBridge.nativeInit()

        loadCalibration()
        configureHighRefreshDisplayAndKeepOn()

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        rotSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
            ?: sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

        val root = FrameLayout(this).apply {
            setBackgroundColor(Color.BLACK)
        }

        glView = GLSurfaceView(this).apply {
            setEGLContextClientVersion(3)
            setEGLConfigChooser(MsaaEglConfigChooser())
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

        addCalibrationOverlay(root)
        setContentView(root)
        applyImmersiveFullscreen()

        if (!hasCameraPermission()) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), REQ_CAMERA)
        }
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
            .maxByOrNull { it.refreshRate }

        if (bestMode != null && bestMode.modeId != curMode.modeId) {
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

    private fun addCalibrationOverlay(root: FrameLayout) {
        val gearBtn = Button(this).apply {
            text = "\u2699"
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 18f)
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.argb(120, 24, 24, 30))
            setPadding(24, 12, 24, 12)
            setOnClickListener {
                panelLayout.visibility =
                    if (panelLayout.visibility == View.VISIBLE) View.GONE else View.VISIBLE
            }
        }
        val gearParams = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT,
            FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.TOP or Gravity.END
        ).apply {
            topMargin = 20
            marginEnd = 24
        }
        root.addView(gearBtn, gearParams)

        panelLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            visibility = View.GONE
            setBackgroundColor(Color.argb(225, 16, 18, 24))
            setPadding(32, 24, 32, 24)
        }

        val fovLabel = TextView(this).apply {
            setTextColor(Color.WHITE)
            text = "Khung Camera (chống chóng mặt): $fovPct%"
        }
        val fovBar = SeekBar(this).apply {
            max = 50 // 50..100
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
            max = 16 // 56..72
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
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
            setPadding(0, 12, 0, 0)
            text = "Mẹo: Nhấn nút Tăng âm lượng (Volume+) để đặt lại khối 3D khi đã đóng hộp."
        }

        panelLayout.addView(fovLabel)
        panelLayout.addView(fovBar)
        panelLayout.addView(ipdLabel)
        panelLayout.addView(ipdBar)
        panelLayout.addView(recenterBtn, btnParams)
        panelLayout.addView(tipText)

        val panelParams = FrameLayout.LayoutParams(
            780,
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
    }

    override fun onPause() {
        recenterRunnable?.let { mainHandler.removeCallbacks(it) }
        sensorManager.unregisterListener(this)
        stopTrackingAndCamera()
        glView.onPause()
        super.onPause()
    }

    override fun onDestroy() {
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
            // getQuaternionFromVector writes [w, x, y, z]
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
                        val preferred = configs.firstOrNull {
                            it.imageSize.height in 720..1080
                        } ?: configs[0]
                        session.cameraConfig = preferred
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
                    Log.i(TAG, "ARCore 6DoF session initialized")
                } catch (t: Throwable) {
                    Log.w(TAG, "ARCore unavailable, falling back to Camera2 + 120Hz 3DoF: ${t.message}")
                    arSession = null
                }
            }

            val session = arSession
            if (session != null) {
                try {
                    session.resume()
                    arNeedsTextureBind = true
                    viewportDirty = true
                    return
                } catch (t: Throwable) {
                    Log.w(TAG, "ARCore resume failed, falling back to Camera2: ${t.message}")
                    session.close()
                    arSession = null
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
        synchronized(sessionLock) {
            fallbackSurfaceTexture = st
            fallbackPreviewSurface = surface
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
                    try {
                        val req = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                            addTarget(surface)
                            set(
                                CaptureRequest.CONTROL_AF_MODE,
                                CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO
                            )
                        }
                        @Suppress("DEPRECATION")
                        device.createCaptureSession(
                            listOf(surface),
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
                                    Log.e(TAG, "Camera2 session configure failed")
                                }
                            },
                            handler
                        )
                    } catch (e: Exception) {
                        Log.e(TAG, "Camera2 createCaptureSession failed: ${e.message}")
                    }
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
                            }
                        } catch (t: Throwable) {
                            Log.w(TAG, "ARCore update exception, switching to Camera2 fallback: ${t.message}")
                            try {
                                session.close()
                            } catch (_: Throwable) {}
                            arSession = null
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
        // Quad corners in standard GL surface coordinates: BL(0,0), BR(1,0), TL(0,1), TR(1,1).
        // Rotate 180 deg if phone was inserted upside-down (ROTATION_270).
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

    private class MsaaEglConfigChooser : GLSurfaceView.EGLConfigChooser {
        override fun chooseConfig(egl: EGL10, display: EGLDisplay): EGLConfig {
            val numConfig = IntArray(1)
            val configs = arrayOfNulls<EGLConfig>(1)

            val msaaAttribs = intArrayOf(
                EGL10.EGL_RED_SIZE, 8,
                EGL10.EGL_GREEN_SIZE, 8,
                EGL10.EGL_BLUE_SIZE, 8,
                EGL10.EGL_DEPTH_SIZE, 16,
                EGL10.EGL_RENDERABLE_TYPE, 4, // EGL_OPENGL_ES2_BIT (compatible with ES3 context)
                EGL10.EGL_SAMPLE_BUFFERS, 1,
                EGL10.EGL_SAMPLES, 4,
                EGL10.EGL_NONE
            )
            if (egl.eglChooseConfig(display, msaaAttribs, configs, 1, numConfig) &&
                numConfig[0] > 0 && configs[0] != null
            ) {
                return configs[0]!!
            }

            val fallbackAttribs = intArrayOf(
                EGL10.EGL_RED_SIZE, 8,
                EGL10.EGL_GREEN_SIZE, 8,
                EGL10.EGL_BLUE_SIZE, 8,
                EGL10.EGL_DEPTH_SIZE, 16,
                EGL10.EGL_RENDERABLE_TYPE, 4,
                EGL10.EGL_NONE
            )
            if (egl.eglChooseConfig(display, fallbackAttribs, configs, 1, numConfig) &&
                numConfig[0] > 0 && configs[0] != null
            ) {
                return configs[0]!!
            }
            throw IllegalStateException("No compatible EGLConfig found")
        }
    }
}
