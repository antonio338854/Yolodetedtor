[app]
title = Detector YOLO
package.name = yolodetector
package.domain = org.tony

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,weights,cfg,names

version = 1.0

requirements = python3,kivy,opencv,numpy

orientation = portrait
fullscreen = 0

android.permissions = CAMERA,INTERNET,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE
android.api = 33
android.minapi = 21
android.ndk = 25b
android.archs = arm64-v8a

[buildozer]
log_level = 2
warn_on_root = 1
