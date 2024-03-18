// ignore_for_file: avoid_print

import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:inference_test/utils/data.dart';
import 'package:inference_test/utils/models.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

DataSet getModelDataSet(Model model) {
  switch (model) {
    case Model.mobilenet:
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
    case Model.mobilenet:
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
      return e.map((e) => e.toDouble() / 255).toList();
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
      case Model.mobilenet:
        return {
          0: List.filled(1 * 1001, 0).reshape([1, 1001])
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

  Future<IsolateInterpreter> _loadTFLiteModel(
    LoadModelOptions options,
  ) async {
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

class ONNXRuntimePerformanceTester extends PerformanceTester {
  @override
  Future<MLInferencePerformanceResult> testPerformance(
      {required LoadModelOptions loadModelOptions}) async {
    print(
        "Loading model: ${loadModelOptions.model} with input precision: ${loadModelOptions.inputPrecision}");
    var assetFileName = ModelsUtil().getModelPath(loadModelOptions.model,
        loadModelOptions.inputPrecision, ModelFormat.onnx);
    final rawAssetFile = await rootBundle.load(assetFileName);
    final bytes = rawAssetFile.buffer.asUint8List();

    OrtEnv.instance.init();
    final sessionOptions = OrtSessionOptions();
    final session = OrtSession.fromBuffer(bytes, sessionOptions);
    final runOptions = OrtRunOptions();

    var ttt = await _getInputs(
        loadModelOptions: loadModelOptions, inputNames: session.inputNames);

    // wait for the model to be ready
    await Future.delayed(const Duration(seconds: 1));

    var fastestTimeMs = double.infinity;
    var slowestTimeMs = 0.0;
    var sum = 0.0;

    print("Running inference");
    for (var data in ttt) {
      var inputOrt = data;
      print("run");
      var start = DateTime.now();
      final outputs =
          await session.runAsync(runOptions, inputOrt, session.outputNames);
      var end = DateTime.now();

      var timeMs = end.difference(start).inMilliseconds;
      print(timeMs);
      if (timeMs < fastestTimeMs) {
        fastestTimeMs = timeMs.toDouble();
      }
      if (timeMs > slowestTimeMs) {
        slowestTimeMs = timeMs.toDouble();
      }
      sum += timeMs;

      for (var key in inputOrt.keys) {
        inputOrt[key]?.release();
      }

      for (var output in outputs ?? []) {
        output?.release();
      }
    }

    print("Inference ready");

    session.release();
    runOptions.release();

    OrtEnv.instance.release();

    return MLInferencePerformanceResult(
        avgPerformanceTimeMs: sum / ttt.length,
        fastestTimeMs: fastestTimeMs,
        slowestTimeMs: slowestTimeMs);
  }

  Future<List<Map<String, OrtValue>>> _getInputs(
      {required LoadModelOptions loadModelOptions,
      required List<String> inputNames}) async {
    var tensors = await _getTensors(loadModelOptions: loadModelOptions);

    var inputName = inputNames[0];
    var result = tensors.map((e) {
      var input = e;
      return {inputName: input};
    }).toList();
    return result;
  }

  Future<List<OrtValueTensor>> _getTensors({
    required LoadModelOptions loadModelOptions,
  }) async {
    print("Getting data");
    var options = FetchImageDataOptions(
        amount: 20, dataset: getModelDataSet(loadModelOptions.model));
    var data = await DataService().fetchImageData(options: options);

    var inputShape = getInputShape(loadModelOptions.model);

    var precisionData =
        formatImageDataToPrecision(data, loadModelOptions.inputPrecision);

    var tensors = precisionData.map((e) {
      if (loadModelOptions.inputPrecision == InputPrecision.uint8) {
        var tensor = OrtValueTensor.createTensorWithDataList(
            Uint8List.fromList(e as List<int>), [1, ...inputShape]);
        return tensor;
      } else {
        var tensor = OrtValueTensor.createTensorWithDataList(
            Float32List.fromList(e as List<double>), [1, ...inputShape]);
        return tensor;
      }
    }).toList();
    print("Data ready");
    return tensors;
  }
}
