// ignore_for_file: prefer_const_constructors_in_immutables, use_key_in_widget_constructors

import 'package:flutter/material.dart';


class ExampleButton extends StatelessWidget {
  final String title;

  ExampleButton({super.key, required this.title});

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      child: Text(title),
      onPressed: () {
        // Define your action here
      }
    );
  }
}



class MyComponent extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Center(
      child: ExampleButton(title: 'Click Me'),
    );
  }
}