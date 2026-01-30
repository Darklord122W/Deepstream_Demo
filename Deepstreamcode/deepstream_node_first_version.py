#!/usr/bin/env python3
import threading
import os
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node

from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

import pyds

MUXER_BATCH_TIMEOUT_USEC = 33000


def _make_bbox(cx, cy, w, h):
    bbox = BoundingBox2D()
    if hasattr(bbox.center, "x"):
        bbox.center.x = float(cx)
        bbox.center.y = float(cy)
        if hasattr(bbox.center, "theta"):
            bbox.center.theta = 0.0
    elif hasattr(bbox.center, "position"):
        bbox.center.position.x = float(cx)
        bbox.center.position.y = float(cy)
        if hasattr(bbox.center, "theta"):
            bbox.center.theta = 0.0
    else:
        raise AttributeError(f"Unsupported bbox.center type: {type(bbox.center)}")

    bbox.size_x = float(w)
    bbox.size_y = float(h)
    return bbox


class DeepStreamUsbNode(Node):
    def __init__(self):
        super().__init__("deepstream_usb_node")

        # Params (similar vibe to your YOLO node)
        self.declare_parameter("video_device", "/dev/video0")
        self.declare_parameter("detections_topic", "/deepstream/detections")
        self.declare_parameter("frame_id", "default_cam")

        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30)

        self.declare_parameter("streammux_width", 1920)
        self.declare_parameter("streammux_height", 1080)

        self.declare_parameter("pgie_config", "dstest1_pgie_config.txt")
        self.declare_parameter("sync", False)

        self.video_device = self.get_parameter("video_device").value
        self.frame_id = self.get_parameter("frame_id").value
        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.fps = int(self.get_parameter("fps").value)

        self.streammux_width = int(self.get_parameter("streammux_width").value)
        self.streammux_height = int(self.get_parameter("streammux_height").value)

        pgie_cfg_param = self.get_parameter("pgie_config").value
        self.pgie_config = self._resolve_pkg_path(pgie_cfg_param)
        self.get_logger().info(f"Using PGIE config: {self.pgie_config}")
        self.sync = bool(self.get_parameter("sync").value)

        det_topic = self.get_parameter("detections_topic").value
        self.det_pub = self.create_publisher(Detection2DArray, det_topic, 10)

        self.get_logger().info(f"DeepStream reading from: {self.video_device}")
        self.get_logger().info(f"Publishing detections: {det_topic}")

        # GStreamer init
        Gst.init(None)
        self.loop = GLib.MainLoop()
        self.loop_thread = threading.Thread(target=self.loop.run, daemon=True)

        self.pipeline = self._build_pipeline()
        self._attach_probe()

        # Start
        self.pipeline.set_state(Gst.State.PLAYING)
        self.loop_thread.start()

    def _resolve_pkg_path(self, path: str) -> str:
        # If user passed an absolute path, trust it
        if os.path.isabs(path):
            return path

        # Otherwise treat it as relative to <share/deepstream_bro/config/>
        pkg_share = get_package_share_directory("deepstream_bro")
        cfg_path = os.path.join(pkg_share, "config", path)

        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"PGIE config not found: '{cfg_path}'. "
                f"Installed config dir: {os.path.join(pkg_share, 'config')}"
            )
        return cfg_path


    def _build_pipeline(self):
        pipeline = Gst.Pipeline.new("ds-pipeline")
        if not pipeline:
            raise RuntimeError("Unable to create pipeline")

        source = Gst.ElementFactory.make("v4l2src", "usb-cam-source")
        caps_v4l2src = Gst.ElementFactory.make("capsfilter", "v4l2src_caps")
        vidconvsrc = Gst.ElementFactory.make("videoconvert", "convertor_src1")
        nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_src2")
        caps_vidconvsrc = Gst.ElementFactory.make("capsfilter", "nvmm_caps")

        streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

        # Use fakesink by default (no window needed)
        sink = Gst.ElementFactory.make("fakesink", "sink")

        elems = [source, caps_v4l2src, vidconvsrc, nvvidconvsrc, caps_vidconvsrc,
                 streammux, pgie, nvvidconv, nvosd, sink]
        if any(e is None for e in elems):
            raise RuntimeError("Failed to create one or more GStreamer elements")

        # Properties (from your script)
        source.set_property("device", self.video_device)

        caps_v4l2src.set_property(
            "caps",
            Gst.Caps.from_string(f"video/x-raw,framerate={self.fps}/1,width={self.width},height={self.height}")
        )
        caps_vidconvsrc.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM)"))

        streammux.set_property("width", self.streammux_width)
        streammux.set_property("height", self.streammux_height)
        streammux.set_property("batch-size", 1)
        streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)

        pgie.set_property("config-file-path", self.pgie_config)

        sink.set_property("sync", self.sync)

        # Add
        for e in elems:
            pipeline.add(e)

        # Link chain until streammux
        if not source.link(caps_v4l2src):
            raise RuntimeError("link failed: source -> caps_v4l2src")
        if not caps_v4l2src.link(vidconvsrc):
            raise RuntimeError("link failed: caps_v4l2src -> videoconvert")
        if not vidconvsrc.link(nvvidconvsrc):
            raise RuntimeError("link failed: videoconvert -> nvvideoconvert")
        if not nvvidconvsrc.link(caps_vidconvsrc):
            raise RuntimeError("link failed: nvvideoconvert -> caps_vidconvsrc")

        # caps_vidconvsrc -> streammux.sink_0 (request pad)
        sinkpad = streammux.request_pad_simple("sink_0")
        srcpad = caps_vidconvsrc.get_static_pad("src")
        if sinkpad is None or srcpad is None:
            raise RuntimeError("Unable to get pads for streammux linking")
        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Failed to link caps_vidconvsrc -> streammux.sink_0")

        # Downstream
        if not streammux.link(pgie):
            raise RuntimeError("link failed: streammux -> pgie")
        if not pgie.link(nvvidconv):
            raise RuntimeError("link failed: pgie -> nvvidconv")
        if not nvvidconv.link(nvosd):
            raise RuntimeError("link failed: nvvidconv -> nvosd")
        if not nvosd.link(sink):
            raise RuntimeError("link failed: nvosd -> sink")

        # Bus watch (simple: print errors to ROS log)
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message, None)

        # Save nvosd for probe
        self._nvosd = nvosd

        return pipeline

    def _on_bus_message(self, bus, message, _):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.get_logger().error(f"GstError: {err} debug={debug}")
        elif t == Gst.MessageType.EOS:
            self.get_logger().info("Gst EOS")

    def _attach_probe(self):
        osdsinkpad = self._nvosd.get_static_pad("sink")
        if not osdsinkpad:
            raise RuntimeError("Unable to get sink pad of nvosd")
        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self._osd_probe, None)

    def _osd_probe(self, pad, info, _):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK

        out = Detection2DArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.frame_id

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                obj = pyds.NvDsObjectMeta.cast(l_obj.data)

                x1 = float(obj.rect_params.left)
                y1 = float(obj.rect_params.top)
                w = float(obj.rect_params.width)
                h = float(obj.rect_params.height)

                cx = x1 + 0.5 * w
                cy = y1 + 0.5 * h

                det = Detection2D()
                det.header = out.header
                det.bbox = _make_bbox(cx, cy, w, h)

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = int(obj.class_id)
                hyp.hypothesis.score = float(obj.confidence)
                det.results.append(hyp)

                out.detections.append(det)

                l_obj = l_obj.next

            l_frame = l_frame.next

        self.det_pub.publish(out)
        return Gst.PadProbeReturn.OK

    def destroy_node(self):
        try:
            if hasattr(self, "pipeline") and self.pipeline is not None:
                self.pipeline.set_state(Gst.State.NULL)
        except Exception:
            pass
        try:
            if hasattr(self, "loop") and self.loop is not None:
                self.loop.quit()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = DeepStreamUsbNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
