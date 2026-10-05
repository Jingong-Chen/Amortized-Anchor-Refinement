#!/bin/bash
# One-shot: reverse-tunnel the viewer to the Quest over USB and open it in the Quest browser.
# (adb reverse makes the headset see our server as localhost => WebXR-allowed secure context.)
ADB=./tools/platform-tools/adb
set -e
$ADB devices | grep -q "device$" || { echo "NO AUTHORIZED DEVICE — enable USB debugging + accept the prompt in the headset"; exit 1; }
$ADB reverse tcp:8123 tcp:8123
echo "reverse tunnel OK"
$ADB shell am start -a android.intent.action.VIEW -d "http://localhost:8123/?f=field40.bin" com.oculus.browser
echo "browser launched — put on the headset, press 'Enter VR'"
