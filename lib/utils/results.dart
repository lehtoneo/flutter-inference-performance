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

  Future sendMobileNetResultsAsync(
      SendResultsOptions<List<num>> options) async {
    var apiOptions = SendResultsApiOptions<List<num>>(options: options);
    return await _sendResultsAsync("$_baseUrl/mobilenet", apiOptions);
  }

  Future sendSSDMobileNetResultsAsync(
      SendResultsOptions<dynamic> options) async {
    var apiOptions = SendResultsApiOptions<dynamic>(options: options);
    return await _sendResultsAsync("$_baseUrl/ssd-mobilenet", apiOptions);
  }

  Future sendDeepLabV3ResultsAsync(
      SendResultsOptions<List<List<num>>> options) async {
    var apiOptions = SendResultsApiOptions<List<List<num>>>(options: options);
    return await _sendResultsAsync("$_baseUrl/deeplabv3", apiOptions);
  }

  Future _sendResultsAsync(String uri, SendResultsApiOptions options) async {
    options.deviceModelName = await _getDeviceModelName();
    print(options);
    return await http.post(
      Uri.parse(uri),
      headers: <String, String>{
        'Content-Type': 'application/json; charset=UTF-8',
      },
      body: jsonEncode(options),
    );
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
    }
  }
}
