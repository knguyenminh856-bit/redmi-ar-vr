import java.net.URI

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

fun gitOutput(vararg args: String, fallback: String): String {
    return try {
        val proc = ProcessBuilder("git", *args)
            .directory(rootProject.projectDir.parentFile)
            .redirectErrorStream(true)
            .start()
        val text = proc.inputStream.bufferedReader().readText().trim()
        if (proc.waitFor() == 0 && text.isNotEmpty()) text else fallback
    } catch (_: Throwable) {
        fallback
    }
}

val gitCommitCount = gitOutput("rev-list", "--count", "HEAD", fallback = "1").toIntOrNull() ?: 1
val gitCommitSha = gitOutput("rev-parse", "HEAD", fallback = "local")

android {
    namespace = "com.minhkhoi.arvr"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.minhkhoi.arvr"
        minSdk = 26
        targetSdk = 34
        versionCode = 100 + gitCommitCount
        versionName = "1.1.$gitCommitCount"

        buildConfigField("String", "GIT_COMMIT_SHA", "\"$gitCommitSha\"")

        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    signingConfigs {
        getByName("debug") {
            storeFile = file("debug-fixed.keystore")
            storePassword = "android"
            keyAlias = "androiddebugkey"
            keyPassword = "android"
        }
    }

    buildTypes {
        getByName("debug") {
            signingConfig = signingConfigs.getByName("debug")
        }
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    buildFeatures {
        buildConfig = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    packaging {
        jniLibs {
            useLegacyPackaging = false
        }
    }

    androidResources {
        noCompress += listOf("task")
    }
}

dependencies {
    implementation("com.google.ar:core:1.45.0")
    implementation("com.google.mediapipe:tasks-vision:1.2.3")
}

val downloadHandLandmarkerModel by tasks.registering {
    val modelFile = layout.projectDirectory.file("src/main/assets/hand_landmarker.task").asFile
    outputs.file(modelFile)
    doLast {
        if (!modelFile.exists() || modelFile.length() < 1_000_000L) {
            modelFile.parentFile.mkdirs()
            val url = URI(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            ).toURL()
            url.openStream().use { input ->
                modelFile.outputStream().use { output -> input.copyTo(output) }
            }
        }
    }
}

tasks.named("preBuild") {
    dependsOn(downloadHandLandmarkerModel)
}
