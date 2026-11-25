package com.minhkhoi.arvr

import android.content.Context
import android.os.Handler
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors

data class SavedRoomInfo(
    val name: String,
    val file: File,
    val sizeKb: Int,
    val dateLabel: String
)

/**
 * Persistent on-phone 3D room cache & `.arroom` v2 (TSDF) + `.obj` storage on a dedicated I/O thread.
 * Uses internal `context.filesDir/rooms` so saved maps ALWAYS persist across app restarts without
 * external-storage permission issues, with automatic background cache (`auto_room_cache.arroom`).
 */
class RoomMapStorage(
    context: Context,
    private val uiHandler: Handler
) {
    companion object {
        const val AUTO_CACHE_FILENAME = "auto_room_cache.arroom"
    }

    private val roomsDir: File = File(context.filesDir, "rooms").apply { mkdirs() }
    private val exportDir: File = File(
        context.getExternalFilesDir(null) ?: context.filesDir,
        "rooms"
    ).apply { mkdirs() }
    private val autoCacheFile: File = File(roomsDir, AUTO_CACHE_FILENAME)

    private val ioExecutor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "RoomDiskIoThread").apply { priority = Thread.NORM_PRIORITY - 1 }
    }

    private val dateFmt = SimpleDateFormat("dd/MM HH:mm", Locale.getDefault())

    fun storageFolderPath(): String = exportDir.absolutePath

    fun hasAutoCache(): Boolean = autoCacheFile.isFile && autoCacheFile.length() >= 32L

    fun autoCacheSummaryText(): String {
        if (!hasAutoCache()) return "Chưa có cache tự lưu trên máy"
        val kb = ((autoCacheFile.length() + 1023L) / 1024L).toInt().coerceAtLeast(1)
        val dt = dateFmt.format(Date(autoCacheFile.lastModified()))
        return "Cache đã lưu trên máy: ${kb} KB ($dt)"
    }

    fun saveAutoCache(onDone: ((Result<Int>) -> Unit)? = null) {
        ioExecutor.execute {
            try {
                val bytes = NativeBridge.nativeSerializeRoom()
                    ?: throw IllegalStateException("Chưa có dữ liệu 3D để lưu cache")
                if (bytes.size < 36) {
                    throw IllegalStateException("Chưa đủ dữ liệu 3D để lưu")
                }
                roomsDir.mkdirs()
                val tmp = File(roomsDir, "$AUTO_CACHE_FILENAME.tmp")
                tmp.writeBytes(bytes)
                if (autoCacheFile.exists()) autoCacheFile.delete()
                if (!tmp.renameTo(autoCacheFile)) {
                    autoCacheFile.writeBytes(bytes)
                    tmp.delete()
                }
                val kb = ((autoCacheFile.length() + 1023L) / 1024L).toInt().coerceAtLeast(1)
                onDone?.let { cb -> uiHandler.post { cb(Result.success(kb)) } }
            } catch (t: Throwable) {
                onDone?.let { cb -> uiHandler.post { cb(Result.failure(t)) } }
            }
        }
    }

    fun loadAutoCache(onDone: (Result<Triple<Int, Int, Int>>) -> Unit) {
        loadRoom(autoCacheFile, onDone)
    }

    fun clearAutoCache(onDone: () -> Unit) {
        ioExecutor.execute {
            try {
                if (autoCacheFile.exists()) autoCacheFile.delete()
                val tmp = File(roomsDir, "$AUTO_CACHE_FILENAME.tmp")
                if (tmp.exists()) tmp.delete()
            } catch (_: Throwable) {
            }
            uiHandler.post { onDone() }
        }
    }

    fun listRooms(onResult: (List<SavedRoomInfo>) -> Unit) {
        ioExecutor.execute {
            val files = roomsDir.listFiles { f ->
                f.isFile && f.name.endsWith(".arroom") && f.name != AUTO_CACHE_FILENAME
            }?.sortedByDescending { it.lastModified() } ?: emptyList()
            val list = files.map { f ->
                SavedRoomInfo(
                    name = f.name.removeSuffix(".arroom"),
                    file = f,
                    sizeKb = ((f.length() + 1023L) / 1024L).toInt().coerceAtLeast(1),
                    dateLabel = dateFmt.format(Date(f.lastModified()))
                )
            }
            uiHandler.post { onResult(list) }
        }
    }

    fun nextAutoRoomName(): String {
        val existing = roomsDir.listFiles { f ->
            f.isFile && f.name.endsWith(".arroom") && f.name != AUTO_CACHE_FILENAME
        }?.map { it.name.removeSuffix(".arroom") }?.toSet() ?: emptySet()
        var idx = 1
        while (existing.contains("phong_$idx")) {
            idx++
        }
        return "phong_$idx"
    }

    fun saveCurrentRoom(name: String, onDone: (Result<SavedRoomInfo>) -> Unit) {
        ioExecutor.execute {
            try {
                val safeName = name.replace(Regex("[^a-zA-Z0-9_-]"), "_").ifBlank { "phong_1" }
                val bytes = NativeBridge.nativeSerializeRoom()
                    ?: throw IllegalStateException("Không thể đóng gói dữ liệu 3D")
                if (bytes.size < 36) {
                    throw IllegalStateException("Chưa có dữ liệu 3D để lưu")
                }
                roomsDir.mkdirs()
                val tmp = File(roomsDir, "$safeName.arroom.tmp")
                val dest = File(roomsDir, "$safeName.arroom")
                tmp.writeBytes(bytes)
                if (dest.exists()) dest.delete()
                if (!tmp.renameTo(dest)) {
                    dest.writeBytes(bytes)
                    tmp.delete()
                }
                // Also sync auto-cache so next startup immediately restores this saved room!
                try {
                    autoCacheFile.writeBytes(bytes)
                } catch (_: Throwable) {}
                val info = SavedRoomInfo(
                    name = safeName,
                    file = dest,
                    sizeKb = ((dest.length() + 1023L) / 1024L).toInt().coerceAtLeast(1),
                    dateLabel = dateFmt.format(Date(dest.lastModified()))
                )
                uiHandler.post { onDone(Result.success(info)) }
            } catch (t: Throwable) {
                uiHandler.post { onDone(Result.failure(t)) }
            }
        }
    }

    fun loadRoom(file: File, onDone: (Result<Triple<Int, Int, Int>>) -> Unit) {
        ioExecutor.execute {
            try {
                if (!file.isFile || file.length() < 32L) {
                    throw IllegalStateException("Không tìm thấy file phòng đã lưu")
                }
                val bytes = file.readBytes()
                val counts = IntArray(3)
                val ok = NativeBridge.nativeLoadRoomBytes(bytes, counts)
                if (!ok) {
                    throw IllegalStateException("File .arroom không hợp lệ hoặc bị hỏng")
                }
                val summary = Triple(counts[0], counts[1], counts[2])
                uiHandler.post { onDone(Result.success(summary)) }
            } catch (t: Throwable) {
                uiHandler.post { onDone(Result.failure(t)) }
            }
        }
    }

    fun exportCurrentRoomObj(baseName: String, onDone: (Result<File>) -> Unit) {
        ioExecutor.execute {
            try {
                val safeName = baseName.replace(Regex("[^a-zA-Z0-9_-]"), "_").ifBlank { "phong_3d" }
                val objText = NativeBridge.nativeExportRoomObj()
                if (objText.isBlank() || !objText.contains("\nv ")) {
                    throw IllegalStateException("Chưa có lưới tam giác 3D để xuất")
                }
                exportDir.mkdirs()
                val outFile = File(exportDir, "$safeName.obj")
                outFile.writeText(objText)
                uiHandler.post { onDone(Result.success(outFile)) }
            } catch (t: Throwable) {
                uiHandler.post { onDone(Result.failure(t)) }
            }
        }
    }

    fun deleteRoom(file: File, onDone: () -> Unit) {
        ioExecutor.execute {
            try {
                file.delete()
            } catch (_: Throwable) {
            }
            uiHandler.post { onDone() }
        }
    }

    fun close() {
        ioExecutor.shutdown()
    }
}
