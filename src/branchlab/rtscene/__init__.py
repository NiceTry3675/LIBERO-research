"""Run-time scene parsing from depth for the RoboDawn relay study.

Turns the depth images of a RoboTwin capture (scripts/replay_capture.py, or the live harness: live.py) into
per-object measurements and already-compared facts that a fast decision model can read as text.

Run-time modules read only what a real robot has: depth and RGB images with camera calibration, its own body
(joint-derived link poses, fingertip pose, gripper opening), the table height and the command history. The
simulator's object poses and segmentation are read only by labels.py, for calibration and scoring;
tests/test_rtscene.py checks that no other module touches them.
"""
