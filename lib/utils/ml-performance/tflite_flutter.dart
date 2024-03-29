// ignore_for_file: avoid_print

import 'dart:typed_data';

import 'package:inference_test/utils/ml-performance/common.dart';
import 'package:inference_test/utils/results.dart';
import 'package:inference_test/utils/data.dart';
import 'package:inference_test/utils/models.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

enum TFLiteDelegate { coreML, nnapi, gpu, cpu }

class TFLitePerformanceTester extends PerformanceTester {
  @override
  Future<MLInferencePerformanceResult> testPerformance({
    required LoadModelOptions loadModelOptions,
  }) async {
    var interpreter = await _loadTFLiteModel(loadModelOptions);
    // create array of [224, 224, 3]
    final inputs = await _getFormattedImageData(
        model: loadModelOptions.model,
        inputPrecision: loadModelOptions.inputPrecision);

    var output = _getOutPutTensor(
        loadModelOptions.model, loadModelOptions.inputPrecision);

    var fastestTimeMs = double.infinity;
    var slowestTimeMs = 0.0;
    var sum = 0.0;

    ResultSender resultSender = ResultSender();

    for (var input in inputs) {
      var startTime = DateTime.now();
      await interpreter.runForMultipleInputs([
        [input]
      ], output);
      var endTime = DateTime.now();

      var timeMs = endTime.difference(startTime).inMilliseconds;

      if (timeMs < fastestTimeMs) {
        fastestTimeMs = timeMs.toDouble();
      }

      if (timeMs > slowestTimeMs) {
        slowestTimeMs = timeMs.toDouble();
      }

      sum += timeMs;

      print("Inference Time: ${timeMs}ms");
    }
    interpreter.close();

    return MLInferencePerformanceResult(
        avgPerformanceTimeMs: sum / inputs.length,
        fastestTimeMs: fastestTimeMs,
        slowestTimeMs: slowestTimeMs);
  }

  dynamic _getOutPutTensor(Model model, InputPrecision inputPrecision) {
    switch (model) {
      case Model.mobilenet_edgetpu:
        return {
          0: List.filled(1 * 1001, 0).reshape([1, 1001])
        };
      case Model.mobilenetv2:
        return {
          0: List.filled(1 * 1000, 0).reshape([1, 1000])
        };
      case Model.ssdMobileNet:
        var s = inputPrecision == InputPrecision.uint8 ? 20 : 10;
        return {
          0: List.filled(1 * s * 4, 0).reshape([1, s, 4]),
          1: List.filled(1 * s, 0).reshape([1, s]),
          2: List.filled(1 * s, 0).reshape([1, s]),
          3: List.filled(1, 0).reshape([1])
        };
      case Model.deeplabv3:
        return {
          0: List.filled(1 * 512 * 512, 0).reshape([1, 512, 512])
        };
      default:
        throw Exception("Unknown model");
    }
  }

  Future<List<dynamic>> _getFormattedImageData({
    required Model model,
    required InputPrecision inputPrecision,
  }) async {
    var options =
        FetchImageDataOptions(amount: 20, dataset: getModelDataSet(model));

    var data = await DataService().fetchImageData(options: options);

    var inputShape = getInputShape(model);

    var precisionData = formatImageDataToPrecision(data, inputPrecision);
    var reshaped = precisionData.map((e) => e.reshape(inputShape)).toList();
    return reshaped;
  }

  Future<IsolateInterpreter> _loadTFLiteModel(LoadModelOptions options) async {
    print(
        "Loading model: ${options.model} with input precision: ${options.inputPrecision}");

    // needed delegate, coreML, nnapi or gpu // ..addDelegate(CoreMlDelegate())
    var interPreterOptions = InterpreterOptions()
      ..useMetalDelegateForIOS = true
      ..useNnApiForAndroid = true;

    var modelPath = ModelsUtil().getModelPath(
        options.model, options.inputPrecision, ModelFormat.tflite);

    final interpreter =
        await Interpreter.fromAsset(modelPath, options: interPreterOptions);

    final isolateInterpreter =
        await IsolateInterpreter.create(address: interpreter.address);

    // lets sleep so that the interpreter is ready to perform inference
    // If this is not done, a error is thrown if inference is performed immediately
    await Future.delayed(const Duration(seconds: 1));

    return isolateInterpreter;
  }
}
