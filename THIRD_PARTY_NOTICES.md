# Third-Party Notices

This project depends on the components below. The version list is locked in
`pyproject.toml`; update this notice whenever a dependency or bundled asset is
changed. Third-party names are used only to identify their software.

## MediaPipe 0.10.35

- Component: `mediapipe`
- Copyright: The MediaPipe Authors
- License: Apache License 2.0
- Source: <https://github.com/google-ai-edge/mediapipe>
- License text: <https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE>
- Additional terms: <https://developers.google.com/edge/mediapipe/legal/tos>

The installed Python distribution includes its license file. Any standalone
application bundle must preserve that license and any NOTICE files shipped by
the distribution.

## OpenCV Python 4.13.0.92

- Component: `opencv-contrib-python`
- Python packaging scripts: MIT License
- OpenCV library: Apache License 2.0
- Source and licensing notes: <https://github.com/opencv/opencv-python>

The wheel contains additional third-party software and corresponding notices,
including codec-related components. A standalone bundle must include the
wheel's `LICENSE.txt` and `LICENSE-3RD-PARTY.txt` files rather than relying only
on this summary.

## NumPy 2.5.0

- Component: `numpy`
- Primary NumPy license: BSD 3-Clause
- License and bundled component notices: <https://numpy.org/doc/stable/license.html>

NumPy ships code under additional compatible licenses, including 0BSD, MIT,
Zlib, and CC0-1.0. A standalone bundle must preserve the complete license files
from the installed NumPy distribution.

## MediaPipe Hand Landmarker model bundle

- Bundled file: `airgesture/resources/models/hand_landmarker.task`
- Contents: `hand_detector.tflite` and `hand_landmarks_detector.tflite`
- Official source used by Google's MediaPipe sample:
  <https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task>
- Official model page:
  <https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker#models>
- SHA-256:
  `FBC2A30080C3C557093B5DDFC334698132EB341044CCEE322CCF8BCF3607CDE1`

The official sample identifies this download as the recommended off-the-shelf
model bundle, but its download endpoint does not provide a separate standalone
license file. Before distributing a commercial or public standalone installer,
the distributor must confirm that its intended redistribution of the model is
covered by Google's then-current terms and retain the relevant model card and
attribution. This is a release approval item, not a claim that the model is
unlicensed.

## Distribution checklist

Before producing an `.exe` or installer:

1. Generate the bundle from the exact locked dependency versions.
2. Copy every license and NOTICE file from the installed distributions into a
   visible `licenses` directory in the final artifact.
3. Include this file, `PRIVACY.md`, and the AirGesture `LICENSE` file.
4. Recheck the bundled model hash and Google's current MediaPipe terms.
5. Have the copyright holder and product owner approve the selected AirGesture
   license and privacy notice. These notices are an engineering inventory, not
   legal advice.
