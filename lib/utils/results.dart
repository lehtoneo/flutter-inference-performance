import 'dart:convert';

import 'package:inference_test/utils/env.dart';
import 'package:inference_test/utils/models.dart';
import 'package:device_info_plus/device_info_plus.dart';

import 'dart:io' show Platform;

import 'package:http/http.dart' as http;

class SendResultsOptions<T> {
  String resultsId = '';
  int inputIndex = 0;
  String inputPrecision = '';
  String library = '';
  late T output;
  double inferenceTimeMs = 0.0;
  String model = '';
  String delegate = '';
}

class SendResultsApiOptions<T> extends SendResultsOptions<T> {
  String platform = Platform.isAndroid ? 'android' : 'ios';
  String deviceModelName = '';
  String frameWork = 'flutter';
}

class ResultSender {
  final String _baseUrl = "$apiEndPoint/results";

  Future sendMobileNetResultsAsync(
      SendResultsApiOptions<List<double>> options) async {
    return await _sendResultsAsync("$_baseUrl/mobilenet", options);
  }

  Future sendSSDMobileNetResultsAsync(
      SendResultsApiOptions<List<List<double>>> options) async {
    return await _sendResultsAsync("$_baseUrl/ssd-mobilenet", options);
  }

  Future sendDeepLabV3ResultsAsync(
      SendResultsApiOptions<List<double>> options) async {
    return await _sendResultsAsync("$_baseUrl/deeplabv3", options);
  }

  Future _sendResultsAsync(String uri, SendResultsApiOptions options) async {
    DeviceInfoPlugin deviceInfo = DeviceInfoPlugin();
    AndroidDeviceInfo androidInfo = await deviceInfo.androidInfo;
    IosDeviceInfo iosInfo = await deviceInfo.iosInfo;

    options.deviceModelName =
        Platform.isAndroid ? androidInfo.model : iosInfo.utsname.machine;

    return await http.post(
      Uri.parse(uri),
      headers: <String, String>{
        'Content-Type': 'application/json; charset=UTF-8',
      },
      body: jsonEncode(options),
    );
  }
}
