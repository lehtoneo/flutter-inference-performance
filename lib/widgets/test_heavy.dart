// ignore_for_file: prefer_const_constructors, unused_element, use_key_in_widget_constructors, library_private_types_in_public_api

import 'dart:isolate';
import 'package:flutter/material.dart';


class ExpensiveWidget extends StatefulWidget {
  @override
  _ExpensiveWidget createState() => _ExpensiveWidget();
}


class _ExpensiveWidget  extends State<ExpensiveWidget> {

  @override
  void initState() {
    super.initState();
    _startHeavyComputation();
  }

  void _startHeavyComputation() async {
    ReceivePort receivePort = ReceivePort();
    await Isolate.spawn(_heavyComputation, receivePort.sendPort);

    receivePort.listen((data) {
      // do something with the result
    });
  }

  static void _heavyComputation(SendPort sendPort) {
    // Perform heavy computation here, e.g., ML inference
    String result = "Computation result";
    sendPort.send(result);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Heavy Computation Example')),
      body: Center(
        child: Text("Heavy Computation Example"),
      ),
    );
  }
}



class NoIsolateExpensiveWidget extends StatefulWidget {
  @override
  _NoIsolateExpensiveWidget createState() => _NoIsolateExpensiveWidget();
}

class _NoIsolateExpensiveWidget extends State<NoIsolateExpensiveWidget> {

  @override
  void initState() {
    super.initState();
    _performComputation();
  }

  void _performComputation() {
    // Perform your heavy computation here
    String result = heavyComputation();

    // do something with the result
  }

  String heavyComputation() {
    // Dummy heavy computation
    int sum = 0;
    // performs a heavy computation
    return "Computation result: $sum";
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('No Isolate Computation Example')),
      body: Center(
        child: Text("No Isolate Computation Example"),
      ),
    );
  }
}

