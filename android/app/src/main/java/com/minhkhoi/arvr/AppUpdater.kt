package com.minhkhoi.arvr

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Handler
import android.provider.Settings
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors

data class UpdateCheckResult(
    val hasUpdate: Boolean,
    val remoteCommit: String,
    val remoteBuildNumber: Int,
    val downloadUrl: String,
    val assetSizeBytes: Long,
    val error: String? = null
)

class AppUpdater(private val uiHandler: Handler) {

    companion object {
        private const val TAG = "AppUpdater"
        private const val RELEASE_API_URL =
            "https://api.github.com/repos/knguyenminh856-bit/redmi-ar-vr/releases/tags/latest-apk"
        private val COMMIT_REGEX = Regex("BuildCommit:\\s*([0-9a-fA-F]{7,40})")
        private val BUILD_NUM_REGEX = Regex("BuildNumber:\\s*(\\d+)")
    }

    private val executor = Executors.newSingleThreadExecutor()

    @Volatile
    var isDownloading = false
        private set

    fun checkForUpdate(onResult: (UpdateCheckResult) -> Unit) {
        executor.execute {
            try {
                val conn = (URL(RELEASE_API_URL).openConnection() as HttpURLConnection).apply {
                    connectTimeout = 10000
                    readTimeout = 10000
                    setRequestProperty("Accept", "application/vnd.github+json")
                    setRequestProperty("Cache-Control", "no-cache")
                    setRequestProperty("User-Agent", "RedmiARVR-Updater")
                }
                val code = conn.responseCode
                val bodyText = try {
                    if (code in 200..299) {
                        conn.inputStream.bufferedReader().use { it.readText() }
                    } else {
                        throw IllegalStateException("HTTP $code")
                    }
                } finally {
                    conn.disconnect()
                }

                val json = JSONObject(bodyText)
                val releaseBody = json.optString("body", "")
                val remoteCommit = COMMIT_REGEX.find(releaseBody)?.groupValues?.get(1) ?: ""
                val remoteBuildNum =
                    BUILD_NUM_REGEX.find(releaseBody)?.groupValues?.get(1)?.toIntOrNull() ?: 0

                val assets = json.optJSONArray("assets")
                var apkUrl = ""
                var apkSize = 0L
                if (assets != null) {
                    for (i in 0 until assets.length()) {
                        val a = assets.getJSONObject(i)
                        val name = a.optString("name", "")
                        if (name.endsWith(".apk", ignoreCase = true) || i == 0) {
                            apkUrl = a.optString("browser_download_url", "")
                            apkSize = a.optLong("size", 0L)
                            if (name.endsWith(".apk", ignoreCase = true)) break
                        }
                    }
                }

                val localSha = BuildConfig.GIT_COMMIT_SHA
                val hasUpdate = remoteCommit.isNotEmpty() &&
                    apkUrl.isNotEmpty() &&
                    !remoteCommit.equals(localSha, ignoreCase = true) &&
                    !localSha.equals("local", ignoreCase = true)

                val res = UpdateCheckResult(
                    hasUpdate = hasUpdate,
                    remoteCommit = remoteCommit,
                    remoteBuildNumber = remoteBuildNum,
                    downloadUrl = apkUrl,
                    assetSizeBytes = apkSize
                )
                uiHandler.post { onResult(res) }
            } catch (t: Throwable) {
                Log.w(TAG, "Update check failed: ${t.message}")
                val res = UpdateCheckResult(
                    hasUpdate = false,
                    remoteCommit = "",
                    remoteBuildNumber = 0,
                    downloadUrl = "",
                    assetSizeBytes = 0L,
                    error = t.message ?: "Lỗi mạng"
                )
                uiHandler.post { onResult(res) }
            }
        }
    }

    /**
     * Returns true if download started, or false if the user was sent to Android Settings
     * to grant "Install unknown apps" permission first.
     */
    fun startDownloadAndInstall(
        activity: Activity,
        downloadUrl: String,
        expectedSize: Long,
        onProgress: (percent: Int, downloadedBytes: Long, totalBytes: Long) -> Unit,
        onReadyToInstall: () -> Unit,
        onError: (String) -> Unit
    ): Boolean {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O &&
            !activity.packageManager.canRequestPackageInstalls()
        ) {
            val intent = Intent(
                Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES,
                Uri.parse("package:${activity.packageName}")
            )
            activity.startActivity(intent)
            return false
        }

        if (isDownloading) return true
        isDownloading = true

        val appCtx = activity.applicationContext
        val finalFile = ApkFileProvider.targetApkFile(appCtx)
        val tmpFile = File(finalFile.parentFile, "${ApkFileProvider.UPDATE_APK_NAME}.tmp")

        executor.execute {
            try {
                finalFile.parentFile?.mkdirs()
                if (tmpFile.exists()) tmpFile.delete()

                val conn = openFollowingRedirects(downloadUrl)
                try {
                    val code = conn.responseCode
                    if (code !in 200..299) {
                        throw IllegalStateException("HTTP $code")
                    }
                    val total = conn.contentLengthLong.let { if (it > 0) it else expectedSize }
                    var downloaded = 0L
                    var lastPct = -1

                    conn.inputStream.use { input ->
                        FileOutputStream(tmpFile).use { output ->
                            val buf = ByteArray(16 * 1024)
                            while (true) {
                                val read = input.read(buf)
                                if (read <= 0) break
                                output.write(buf, 0, read)
                                downloaded += read
                                val pct = if (total > 0) ((downloaded * 100L) / total).toInt().coerceIn(0, 100) else 0
                                if (pct != lastPct) {
                                    lastPct = pct
                                    uiHandler.post { onProgress(pct, downloaded, total) }
                                }
                            }
                            output.flush()
                        }
                    }

                    if (tmpFile.length() < 1_000_000L || !hasZipMagic(tmpFile)) {
                        throw IllegalStateException("Tệp APK tải về không hợp lệ")
                    }
                    if (finalFile.exists()) finalFile.delete()
                    if (!tmpFile.renameTo(finalFile)) {
                        tmpFile.copyTo(finalFile, overwrite = true)
                        tmpFile.delete()
                    }
                } finally {
                    conn.disconnect()
                }

                uiHandler.post {
                    onReadyToInstall()
                    launchInstaller(activity)
                }
            } catch (t: Throwable) {
                Log.e(TAG, "APK update failed: ${t.message}")
                if (tmpFile.exists()) tmpFile.delete()
                uiHandler.post { onError(t.message ?: "Lỗi tải bản cập nhật") }
            } finally {
                isDownloading = false
            }
        }
        return true
    }

    private fun openFollowingRedirects(initialUrl: String): HttpURLConnection {
        var currentUrl = initialUrl
        var hops = 0
        while (true) {
            val conn = (URL(currentUrl).openConnection() as HttpURLConnection).apply {
                instanceFollowRedirects = false
                connectTimeout = 15000
                readTimeout = 30000
                setRequestProperty("User-Agent", "RedmiARVR-Updater")
            }
            val code = conn.responseCode
            if (code in listOf(301, 302, 303, 307, 308)) {
                val loc = conn.getHeaderField("Location") ?: return conn
                conn.disconnect()
                currentUrl = URL(URL(currentUrl), loc).toString()
                if (++hops > 6) throw IllegalStateException("Quá nhiều lần chuyển hướng")
            } else {
                return conn
            }
        }
    }

    private fun hasZipMagic(file: File): Boolean {
        return try {
            file.inputStream().use { ins ->
                val header = ByteArray(4)
                ins.read(header) == 4 &&
                    header[0] == 0x50.toByte() &&
                    header[1] == 0x4B.toByte() &&
                    header[2] == 0x03.toByte() &&
                    header[3] == 0x04.toByte()
            }
        } catch (_: Throwable) {
            false
        }
    }

    private fun launchInstaller(activity: Activity) {
        val apkUri = ApkFileProvider.contentUri(activity)
        val intent = Intent(Intent.ACTION_VIEW).apply {
            setDataAndType(apkUri, "application/vnd.android.package-archive")
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        activity.startActivity(intent)
    }

    fun close() {
        executor.shutdown()
    }
}
