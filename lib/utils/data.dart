import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:inference_test/utils/env.dart';
import 'package:permission_handler/permission_handler.dart';

class Buffer {
  final String type;
  final List<int> data;

  Buffer({required this.type, required this.data});

  factory Buffer.fromJson(Map<String, dynamic> json) {
    return Buffer(
      type: json['type'],
      data: List<int>.from(json['data']),
    );
  }
}

class FetchImagesQueryData {
  final String id;
  final Buffer buffer;
  final Buffer rawImageBuffer;

  FetchImagesQueryData(
      {required this.id, required this.buffer, required this.rawImageBuffer});

  factory FetchImagesQueryData.fromJson(Map<String, dynamic> json) {
    return FetchImagesQueryData(
      id: json['id'],
      buffer: Buffer.fromJson(json['buffer']),
      rawImageBuffer: Buffer.fromJson(json['rawImageBuffer']),
    );
  }
}

enum DataSet { coco, imagenet, ade20k }

String _getDataSetPath(DataSet dataSet) {
  switch (dataSet) {
    case DataSet.coco:
      return 'coco';
    case DataSet.ade20k:
      return 'ade20k';
    case DataSet.imagenet:
      return 'imagenet';
    default:
      return 'Unknown';
  }
}

class FetchImageDataOptions {
  final int amount;
  final DataSet dataset;
  int skip;

  FetchImageDataOptions(
      {required this.amount, required this.dataset, this.skip = 0});
}

class _FetchImageDataApiOptions extends FetchImageDataOptions {
  final int skip;

  _FetchImageDataApiOptions(
      {required int amount, required this.skip, required DataSet dataset})
      : super(amount: amount, dataset: dataset);
}

class DataService {
  Future<List<FetchImagesQueryData>> fetchImageData({
    required FetchImageDataOptions options,
  }) async {
    // need to fetch in batches as the request response payload is too large in some cases
    var batchSize = 50;

    if (options.amount <= batchSize) {
      return await _fetchImageDataFromApi(
        options: _FetchImageDataApiOptions(
          amount: options.amount,
          skip: options.skip,
          dataset: options.dataset,
        ),
      );
    }

    var data = <FetchImagesQueryData>[];

    for (var i = 0; i < options.amount; i += batchSize) {
      var result = await _fetchImageDataFromApi(
        options: _FetchImageDataApiOptions(
          amount: batchSize,
          skip: options.skip,
          dataset: options.dataset,
        ),
      );
      data.addAll(result);
    }

    return data;
  }

  Future<List<FetchImagesQueryData>> _fetchImageDataFromApi({
    required _FetchImageDataApiOptions options,
  }) async {
    final int amount = options.amount;
    if (amount <= 0) {
      throw Exception('Amount should be greater than 0');
    }
    if (amount > 300) {
      throw Exception('Amount should be less than or equal to 300');
    }

    var skip = options.skip;
    final String datasetValue = _getDataSetPath(options.dataset);

    final url = Uri.parse(
        '$apiEndPoint/api/data/$datasetValue?amount=$amount&skip=$skip');

    final response = await http.get(url);

    if (response.statusCode == 200) {
      List<dynamic> jsonList = json.decode(response.body);
      return jsonList
          .map((json) => FetchImagesQueryData.fromJson(json))
          .toList();
    } else {
      throw Exception('Failed to load data');
    }
  }

  static Future<bool> isReachable() async {
    var t = await Permission.nearbyWifiDevices.request();

    var url = Uri.parse("$apiEndPoint/health");

    try {
      var response = await http.get(url);
      return response.statusCode == 200;
    } catch (e) {
      print(e.toString());
      return false;
    }
  }
}
