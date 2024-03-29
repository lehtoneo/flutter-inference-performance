import 'package:tflite_flutter/tflite_flutter.dart';

enum Model { mobilenetv2, mobilenet_edgetpu, ssdMobileNet, deeplabv3 }

enum InputPrecision { uint8, float32 }

enum ModelFormat { tflite, onnx }

class LoadModelOptions {
  final Model model;
  final InputPrecision inputPrecision;

  LoadModelOptions({required this.model, required this.inputPrecision});
}

class ModelsUtil {
  String _getInputPrecisionString(InputPrecision inputPrecision) {
    switch (inputPrecision) {
      case InputPrecision.uint8:
        return 'uint8';
      case InputPrecision.float32:
        return 'float32';
      default:
        return 'Unknown';
    }
  }

  String _getModelFormatString(ModelFormat format) {
    switch (format) {
      case ModelFormat.tflite:
        return 'tflite';
      case ModelFormat.onnx:
        return 'onnx';
      default:
        return 'Unknown';
    }
  }

  String getModelPath(
      Model model, InputPrecision inputPrecision, ModelFormat format) {
    var precision = _getInputPrecisionString(inputPrecision);
    var modelFormat = _getModelFormatString(format);
    switch (model) {
      case Model.mobilenetv2:
        return 'assets/$modelFormat/mobilenetv2_$precision.$modelFormat';
      case Model.mobilenet_edgetpu:
        return 'assets/$modelFormat/mobilenet_edgetpu_224_1.0_$precision.$modelFormat';
      case Model.ssdMobileNet:
        return 'assets/$modelFormat/ssd_mobilenet_v2_300_$precision.$modelFormat';
      case Model.deeplabv3:
        return 'assets/$modelFormat/deeplabv3_mnv2_ade20k_$precision.$modelFormat';
      default:
        throw Exception("Unknown model");
    }
  }
}
