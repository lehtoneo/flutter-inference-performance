import 'package:flutter/material.dart';
import 'package:inference_test/utils/ml-performance.dart';

class InferenceResultDisplay extends StatelessWidget {
  final MLInferencePerformanceResult? performanceResult;

  const InferenceResultDisplay({Key? key, this.performanceResult})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    if (performanceResult == null) {
      return const Text('No results to display.');
    }

    return Column(
      children: [
        Text(
            'Average performance time: ${performanceResult!.avgPerformanceTimeMs} ms'),
        Text('Fastest time: ${performanceResult!.fastestTimeMs} ms'),
        Text('Slowest time: ${performanceResult!.slowestTimeMs} ms'),
      ],
    );
  }
}
