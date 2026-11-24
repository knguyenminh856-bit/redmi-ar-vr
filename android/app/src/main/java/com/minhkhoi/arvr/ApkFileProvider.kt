package com.minhkhoi.arvr

import android.content.ContentProvider
import android.content.ContentValues
import android.content.Context
import android.database.Cursor
import android.database.MatrixCursor
import android.net.Uri
import android.os.ParcelFileDescriptor
import android.provider.OpenableColumns
import java.io.File
import java.io.FileNotFoundException

class ApkFileProvider : ContentProvider() {

    companion object {
        const val UPDATE_APK_NAME = "RedmiNote14-ARVR-update.apk"

        fun targetApkFile(context: Context): File {
            val dir = context.externalCacheDir ?: context.cacheDir
            return File(dir, UPDATE_APK_NAME)
        }

        fun contentUri(context: Context): Uri {
            return Uri.parse("content://${context.packageName}.apkprovider/update.apk")
        }
    }

    override fun onCreate(): Boolean = true

    override fun getType(uri: Uri): String = "application/vnd.android.package-archive"

    override fun query(
        uri: Uri,
        projection: Array<out String>?,
        selection: String?,
        selectionArgs: Array<out String>?,
        sortOrder: String?
    ): Cursor? {
        val ctx = context ?: return null
        val file = targetApkFile(ctx)
        val cols = projection ?: arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE)
        val row = Array<Any?>(cols.size) { null }
        for (i in cols.indices) {
            when (cols[i]) {
                OpenableColumns.DISPLAY_NAME -> row[i] = UPDATE_APK_NAME
                OpenableColumns.SIZE -> row[i] = if (file.exists()) file.length() else 0L
            }
        }
        return MatrixCursor(cols, 1).apply { addRow(row) }
    }

    override fun openFile(uri: Uri, mode: String): ParcelFileDescriptor {
        val ctx = context ?: throw FileNotFoundException("Missing provider context")
        val file = targetApkFile(ctx)
        if (!file.exists()) {
            throw FileNotFoundException("Update APK not found: ${file.absolutePath}")
        }
        return ParcelFileDescriptor.open(file, ParcelFileDescriptor.MODE_READ_ONLY)
    }

    override fun insert(uri: Uri, values: ContentValues?): Uri? = null
    override fun delete(uri: Uri, selection: String?, selectionArgs: Array<out String>?): Int = 0
    override fun update(
        uri: Uri,
        values: ContentValues?,
        selection: String?,
        selectionArgs: Array<out String>?
    ): Int = 0
}
