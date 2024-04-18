import 'dart:convert';

import 'package:inference_test/utils/env.dart';
import 'package:inference_test/utils/models.dart';
import 'package:device_info_plus/device_info_plus.dart';

import 'dart:io' show Platform;

import 'package:http/http.dart' as http;

class SendResultsOptions<T> {
  SendResultsOptions(
      {required this.resultsId,
      required this.inputIndex,
      required this.precision,
      required this.library,
      required this.output,
      required this.inferenceTimeMs,
      required this.model,
      required this.delegate});

  String resultsId = '';
  int inputIndex = 0;
  InputPrecision precision = InputPrecision.uint8;
  String library = '';
  late T output;
  double inferenceTimeMs = 0.0;
  Model model = Model.mobilenet_edgetpu;
  DelegateOption delegate = DelegateOption.cpu;
}

class SendResultsApiOptions<T> extends SendResultsOptions<T> {
  SendResultsApiOptions({
    required SendResultsOptions<T> options,
  }) : super(
            resultsId: options.resultsId,
            inputIndex: options.inputIndex,
            precision: options.precision,
            library: options.library,
            output: options.output,
            inferenceTimeMs: options.inferenceTimeMs,
            model: options.model,
            delegate: options.delegate);

  String platform = Platform.isAndroid ? 'android' : 'ios';
  String deviceModelName = '';
  String frameWork = 'flutter';

  Map<String, dynamic> toJson() {
    return {
      'resultsId': resultsId,
      'inputIndex': inputIndex,
      'precision': precision.name,
      'library': library,
      'output': output,
      'inferenceTimeMs': inferenceTimeMs,
      'model': model.name,
      'delegate': delegate.name,
      'platform': platform,
      'deviceModelName': deviceModelName,
      'frameWork': frameWork,
    };
  }
}

class ResultSender {
  final String _baseUrl = "$apiEndPoint/api/results";

  Future sendMultipleResultsAsync(
      Model model, List<SendResultsOptions<dynamic>> options) async {
    var i = 0;
    var pathName = getModelApiPathName(model);
    while (i < options.length) {
      var s = options[i];
      var apiOptions = SendResultsApiOptions(options: s);
      await _sendResultsAsync("$_baseUrl/$pathName", apiOptions);
      i++;
    }
  }

  Future _sendResultsAsync(String uri, SendResultsApiOptions options) async {
    options.deviceModelName = await _getDeviceModelName();
    print(options);
    var result = await http.post(
      Uri.parse(uri),
      headers: <String, String>{
        'Content-Type': 'application/json; charset=UTF-8',
      },
      body: jsonEncode(options),
    );
    if (result.statusCode != 200) {
      print("Failed to send results");
    }
    return result;
  }

  Future<bool> hasResults(SendResultsApiOptions options) async {
    options.deviceModelName = await _getDeviceModelName();
    var result = await http.post(
      Uri.parse("$_baseUrl/has-results"),
      headers: <String, String>{
        'Content-Type': 'application/json; charset=UTF-8',
      },
      body: jsonEncode(options),
    );
    if (result.statusCode == 404) {
      return false;
    }
    return true;
  }

  String getModelApiPathName(Model model) {
    switch (model) {
      case Model.mobilenet_edgetpu:
        return 'mobilenet';
      case Model.mobilenetv2:
        return 'mobilenet';
      case Model.ssd_mobilenet:
        return 'ssd-mobilenet';
      case Model.deeplabv3:
        return 'deeplabv3';
      default:
        throw Exception("Unknown model");
    }
  }

  Future<String> _getDeviceModelName() async {
    DeviceInfoPlugin deviceInfo = DeviceInfoPlugin();

    switch (Platform.isAndroid) {
      case true:
        AndroidDeviceInfo androidInfo = await deviceInfo.androidInfo;
        return androidInfo.device;
      case false:
        IosDeviceInfo iosInfo = await deviceInfo.iosInfo;
        return iosInfo.utsname.machine;
      default:
        throw Exception("Unknown platform");
    }
  }
}
