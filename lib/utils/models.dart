enum Model { mobilenet, mobilenet_edgetpu, ssd_mobilenet, deeplabv3 }

enum InputPrecision {
  uint8,
  float32,
}

enum ModelFormat { tflite, onnx }

enum DelegateOption { core_ml, nnapi, gpu, metal, cpu, xxnpack }

class LoadModelOptions {
  final Model model;
  final InputPrecision inputPrecision;
  final DelegateOption delegate;

  LoadModelOptions(
      {required this.model,
      required this.inputPrecision,
      required this.delegate});
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
      case Model.mobilenet:
        return 'assets/$modelFormat/mobilenetv2_$precision.$modelFormat';
      case Model.mobilenet_edgetpu:
        return 'assets/$modelFormat/mobilenet_edgetpu_224_1.0_$precision.$modelFormat';
      case Model.ssd_mobilenet:
        return 'assets/$modelFormat/ssd_mobilenet_v2_300_$precision.$modelFormat';
      case Model.deeplabv3:
        return 'assets/$modelFormat/deeplabv3_mnv2_ade20k_$precision.$modelFormat';
      default:
        throw Exception("Unknown model");
    }
  }
}
