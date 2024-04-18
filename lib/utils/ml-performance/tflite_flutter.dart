// ignore_for_file: avoid_print

import 'package:inference_test/utils/ml-performance/common.dart';
import 'package:inference_test/utils/data.dart';
import 'package:inference_test/utils/models.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

enum TFLiteDelegate { coreML, nnapi, gpu, cpu }

class TFLitePerformanceTester
    extends PerformanceTester<Interpreter, dynamic, dynamic> {
  @override
  String get libraryName => "tflite_flutter";

  @override
  List<DelegateOption> getLibraryDelegateOptions() {
    return DelegateOption.values;
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
      case Model.ssd_mobilenet:
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

  Future<List<dynamic>> _getFormattedImageData(
      {required Model model,
      required InputPrecision inputPrecision,
      required int skip,
      required int batchSize}) async {
    print("Fetching data for model: ${model.name}");

    var options = FetchImageDataOptions(
        amount: batchSize, dataset: getModelDataSet(model), skip: skip);

    var data = await DataService().fetchImageData(options: options);
    print("Data fetched");

    return _formatImageData(data, inputPrecision, model);
  }

  List<dynamic> _formatImageData(List<FetchImagesQueryData> data,
      InputPrecision inputPrecision, Model model) {
    var inputShape = getInputShape(model);

    var precisionData = formatImageDataToPrecision(data, inputPrecision, model);
    var reshaped = precisionData.map((e) => e.reshape(inputShape)).toList();
    print("Data formatted");
    return reshaped;
  }

  Future<Interpreter> _loadTFLiteModel(LoadModelOptions options) async {
    print(
        "Loading model: ${options.model} with input precision: ${options.inputPrecision}");

    var interPreterOptions = _getInterpreterOptions(options);

    var modelPath = ModelsUtil().getModelPath(
        options.model, options.inputPrecision, ModelFormat.tflite);

    final interpreter =
        await Interpreter.fromAsset(modelPath, options: interPreterOptions);

    // lets sleep so that the interpreter is ready to perform inference
    // If this is not done, a error is thrown if inference is performed immediately
    await Future.delayed(const Duration(seconds: 1));

    return interpreter;
  }

  InterpreterOptions _getInterpreterOptions(LoadModelOptions loadModelOptions) {
    var interPreterOptions = InterpreterOptions();

    // needed delegate, coreML, nnapi or gpu // ..addDelegate(CoreMlDelegate())
    if (loadModelOptions.delegate == DelegateOption.core_ml) {
      interPreterOptions.addDelegate(CoreMlDelegate());
    } else if (loadModelOptions.delegate == DelegateOption.nnapi) {
      interPreterOptions.useNnApiForAndroid = true;
    } else if (loadModelOptions.delegate == DelegateOption.metal) {
      interPreterOptions.useMetalDelegateForIOS = true;
    } else if (loadModelOptions.delegate == DelegateOption.gpu) {
      interPreterOptions.addDelegate(GpuDelegateV2());
    } else if (loadModelOptions.delegate == DelegateOption.xnnpack) {
      interPreterOptions.addDelegate(XNNPackDelegate());
    } else if (loadModelOptions.delegate == DelegateOption.cpu) {
    } else {
      throw Exception("Unknown delegate");
    }

    return interPreterOptions;
  }

  @override
  Future<Interpreter> loadModelAsync(LoadModelOptions loadModelOptions) {
    // TODO: implement loadModelAsync
    return _loadTFLiteModel(loadModelOptions);
  }

  @override
  Future<List> getFormattedInputs(
      {required LoadModelOptions loadModelOptions,
      required Interpreter model,
      required int skip,
      required int batchSize}) {
    return _getFormattedImageData(
        model: loadModelOptions.model,
        inputPrecision: loadModelOptions.inputPrecision,
        skip: skip,
        batchSize: batchSize);
  }

  @override
  formatMobileNetEdgeTpuOutput(out) {
    return out[0][0];
  }

  @override
  Future runInference(
      {required model,
      required input,
      required outputTensor,
      required LoadModelOptions loadModelOptions}) async {
    var output = _getOutPutTensor(
        loadModelOptions.model, loadModelOptions.inputPrecision);
    model.runForMultipleInputs([input], output);
    return output;
  }

  @override
  Future<void> closeModel(Interpreter model) {
    model.close();
    return Future.value();
  }

  @override
  formatDeepLabV3Output(out) {
    return out[0][0];
  }

  @override
  formatMobileNetV2Output(out) {
    return formatMobileNetEdgeTpuOutput(out);
  }

  @override
  formatSSDMobileNetOutput(output) {
    return [output[0][0][0][0], output[1][0], output[2][0], output[3]];
  }

  @override
  Future onAfterInputRun({required input, required output}) async {}

  @override
  dynamic getOutputTensor({required LoadModelOptions loadModelOptions}) async {
    return _getOutPutTensor(
        loadModelOptions.model, loadModelOptions.inputPrecision);
  }
}
