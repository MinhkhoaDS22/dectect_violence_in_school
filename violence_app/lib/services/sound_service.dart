// lib/services/sound_service.dart
// ignore_for_file: avoid_web_libraries_in_flutter
import 'dart:js' as js;
import 'package:shared_preferences/shared_preferences.dart';

/// Dịch vụ phát âm thanh cảnh báo bạo lực qua Web Audio API.
/// Dùng OscillatorNode — không cần file âm thanh.
class SoundService {
  SoundService._();
  static final SoundService instance = SoundService._();

  bool _enabled = true;
  double _volume = 0.80; // 0.0 → 1.0

  bool get enabled => _enabled;
  double get volume => _volume;

  // ── Khởi tạo từ SharedPreferences ──────────────────────
  Future<void> init() async {
    final p = await SharedPreferences.getInstance();
    _enabled = p.getBool('sound_enabled') ?? true;
    _volume = p.getDouble('sound_volume') ?? 0.80;
  }

  Future<void> setEnabled(bool val) async {
    _enabled = val;
    final p = await SharedPreferences.getInstance();
    await p.setBool('sound_enabled', val);
  }

  Future<void> setVolume(double val) async {
    _volume = val.clamp(0.0, 1.0);
    final p = await SharedPreferences.getInstance();
    await p.setDouble('sound_volume', _volume);
  }

  // ── Phát cảnh báo đầy đủ (5 tiếng beep leo thang) ──────
  // Giống logic winsound trong front_end_test.py nhưng qua Web Audio API
  void playAlert() {
    if (!_enabled) return;
    _playSequence([
      [1000, 0.00, 0.38],
      [1500, 0.43, 0.28],
      [1000, 0.76, 0.38],
      [1500, 1.19, 0.28],
      [2200, 1.52, 0.60],
    ], _volume);
  }

  // ── Test beep ngắn khi kéo slider ───────────────────────
  void playTestBeep() {
    _playSequence([
      [1000, 0.0, 0.20],
    ], _volume * 0.6);
  }

  // ── Core: chạy Web Audio API qua JS interop ─────────────
  static void _playSequence(List<List<num>> beeps, double masterVol) {
    final beepsJs = beeps
        .map((b) => '[${b[0]}, ${b[1]}, ${b[2]}]')
        .join(',');

    final script = '''
      (function() {
        try {
          var AudioCtx = window.AudioContext || window.webkitAudioContext;
          if (!AudioCtx) return;
          var ctx = new AudioCtx();

          // Master volume
          var master = ctx.createGain();
          master.gain.value = ${masterVol.toStringAsFixed(3)};
          master.connect(ctx.destination);

          var beeps = [$beepsJs];

          beeps.forEach(function(b) {
            var freq   = b[0];
            var offset = b[1];
            var dur    = b[2];

            var osc  = ctx.createOscillator();
            var gain = ctx.createGain();
            osc.connect(gain);
            gain.connect(master);

            osc.type = 'sine';
            osc.frequency.value = freq;

            var t0 = ctx.currentTime + offset;
            // Attack → sustain → release
            gain.gain.setValueAtTime(0, t0);
            gain.gain.linearRampToValueAtTime(1.0, t0 + 0.02);
            gain.gain.setValueAtTime(1.0, t0 + dur - 0.06);
            gain.gain.linearRampToValueAtTime(0, t0 + dur);

            osc.start(t0);
            osc.stop(t0 + dur + 0.01);
          });
        } catch(e) {
          console.warn('[SafeWatch] Sound error:', e);
        }
      })();
    ''';

    js.context.callMethod('eval', [script]);
  }
}
