import 'dart:async';

Future<T> timeout<T>(Future<T> future, Duration duration,
    {String message = 'Operation timed out'}) {
  return future.timeout(duration, onTimeout: () {
    throw TimeoutException(message);
  });
}
