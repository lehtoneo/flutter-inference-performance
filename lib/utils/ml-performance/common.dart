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
        DelegateOption.xnnpack
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
      case Model.mobilenet_edgetpu || Model.mobilenetv2:
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
      case Model.mobilenet_edgetpu || Model.mobilenetv2:
        return [224, 224, 3];
      case Model.ssd_mobilenet:
        return [300, 300, 3];
      case Model.deeplabv3:
        return [512, 512, 3];
      default:
        throw Exception("Unknown model");
    }
  }

  List<List<num>> formatImageDataToPrecision(List<FetchImagesQueryData> data,
      InputPrecision inputPrecision, Model model) {
    var asUint8 = data.map((e) => e.rawImageBuffer.data).toList();
    if (inputPrecision == InputPrecision.uint8) {
      return asUint8;
    } else {
      ;
      switch (model) {
        case Model.mobilenetv2:
          return asUint8.map((e) {
            return e
                .map((e) => 0.007843137718737125 * (e.toDouble() - 127))
                .toList();
          }).toList();
        case Model.mobilenet_edgetpu:
          var asDoubles = asUint8.map((e) {
            return e
                .map((e) => 0.007874015718698502 * (e.toDouble() - 128))
                .toList();
          }).toList();
          return asDoubles;
        case Model.ssd_mobilenet:
          return asUint8.map((e) {
            return e.map((e) => 0.0078125 * (e.toDouble() - 128)).toList();
          }).toList();
        case Model.deeplabv3:
          return asUint8.map((e) {
            return e.map((e) => 0.0078125 * (e.toDouble() - 128)).toList();
          }).toList();
        default:
          throw Exception("Unknown model");
      }
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
