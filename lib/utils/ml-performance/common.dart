import 'package:inference_test/utils/data.dart';
import 'package:inference_test/utils/models.dart';

DataSet getModelDataSet(Model model) {
  switch (model) {
    case Model.mobilenet_edgetpu || Model.mobilenetv2:
      return DataSet.imagenet;
    case Model.ssdMobileNet:
      return DataSet.coco;
    case Model.deeplabv3:
      return DataSet.ade20k;
    default:
      throw Exception("Unknown model");
  }
}

List<int> getInputShape(Model model) {
  switch (model) {
    case Model.mobilenet_edgetpu || Model.mobilenetv2:
      return [224, 224, 3];
    case Model.ssdMobileNet:
      return [300, 300, 3];
    case Model.deeplabv3:
      return [512, 512, 3];
    default:
      throw Exception("Unknown model");
  }
}

List<List<num>> formatImageDataToPrecision(
    List<FetchImagesQueryData> data, InputPrecision inputPrecision) {
  if (inputPrecision == InputPrecision.uint8) {
    var asUint8 = data.map((e) => e.rawImageBuffer.data).toList();
    return asUint8;
  } else {
    var buffers = data.map((e) => e.rawImageBuffer.data).toList();
    var asDoubles = buffers.map((e) {
      return e.map((e) => e.toDouble() / 127 - 1).toList();
    }).toList();
    return asDoubles;
  }
}

class MLInferencePerformanceResult {
  final double avgPerformanceTimeMs;
  final double fastestTimeMs;
  final double slowestTimeMs;

  MLInferencePerformanceResult(
      {required this.avgPerformanceTimeMs,
      required this.fastestTimeMs,
      required this.slowestTimeMs});
}

abstract class PerformanceTester {
  Future<MLInferencePerformanceResult> testPerformance({
    required LoadModelOptions loadModelOptions,
  });
}
