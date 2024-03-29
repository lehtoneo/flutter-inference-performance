import 'dart:io';

import 'package:inference_test/utils/data.dart';
import 'package:inference_test/utils/models.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class MLInferencePerformanceResult {
  final double avgPerformanceTimeMs;
  final double fastestTimeMs;
  final double slowestTimeMs;

  MLInferencePerformanceResult(
      {required this.avgPerformanceTimeMs,
      required this.fastestTimeMs,
      required this.slowestTimeMs});
}

class PerformanceTesterCommon {
  List<DelegateOption> _getCurrentPlatformDelegateOptions() {
    if (Platform.isAndroid) {
      return [
        DelegateOption.nnapi,
        DelegateOption.gpu,
        DelegateOption.cpu,
        DelegateOption.xxnpack
      ];
    } else if (Platform.isIOS) {
      // ios
      return [
        DelegateOption.core_ml,
        DelegateOption.gpu,
        DelegateOption.metal,
        DelegateOption.cpu
      ];
    } else {
      throw Exception("Unknown platform");
    }
  }

  DataSet getModelDataSet(Model model) {
    switch (model) {
      case Model.mobilenet_edgetpu || Model.mobilenet:
        return DataSet.imagenet;
      case Model.ssd_mobilenet:
        return DataSet.coco;
      case Model.deeplabv3:
        return DataSet.ade20k;
      default:
        throw Exception("Unknown model");
    }
  }

  List<int> getInputShape(Model model) {
    switch (model) {
      case Model.mobilenet_edgetpu || Model.mobilenet:
        return [224, 224, 3];
      case Model.ssd_mobilenet:
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
}

abstract class PerformanceTester extends PerformanceTesterCommon {
  String get libraryName;

  Future<MLInferencePerformanceResult> testPerformance({
    required LoadModelOptions loadModelOptions,
  });

  List<DelegateOption> getLibraryDelegateOptions();

  List<DelegateOption> getDelegateOptions() {
    var platformDelegateOptions = _getCurrentPlatformDelegateOptions();
    var libraryDelegateOptions = getLibraryDelegateOptions();
    return platformDelegateOptions
        .where((element) => libraryDelegateOptions.contains(element))
        .toList();
  }
}
