// lib/widgets/sound_control_bar.dart
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../theme/app_theme.dart';
import '../services/sound_service.dart';

/// Widget điều khiển âm thanh cảnh báo.
/// Gồm: nút tắt/mở + thanh trượt âm lượng + nút test.
class SoundControlBar extends StatefulWidget {
  const SoundControlBar({super.key});

  @override
  State<SoundControlBar> createState() => _SoundControlBarState();
}

class _SoundControlBarState extends State<SoundControlBar>
    with SingleTickerProviderStateMixin {
  final _snd = SoundService.instance;
  late AnimationController _pulseCtrl;

  @override
  void initState() {
    super.initState();
    _pulseCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 600),
    );
  }

  @override
  void dispose() {
    _pulseCtrl.dispose();
    super.dispose();
  }

  void _toggle() async {
    await _snd.setEnabled(!_snd.enabled);
    setState(() {});
  }

  void _onVolumeChange(double v) async {
    await _snd.setVolume(v);
    setState(() {});
  }

  void _onVolumeEnd(double v) {
    // Phát beep test khi thả slider
    _snd.playTestBeep();
  }

  void _testSound() {
    _snd.playAlert();
    _pulseCtrl.forward(from: 0).then((_) => _pulseCtrl.reverse());
  }

  @override
  Widget build(BuildContext context) {
    final enabled = _snd.enabled;
    final vol = _snd.volume;

    return AnimatedContainer(
      duration: const Duration(milliseconds: 250),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: enabled
            ? AppColors.primary.withOpacity(.07)
            : AppColors.surfaceAlt,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: enabled
              ? AppColors.primary.withOpacity(.25)
              : AppColors.divider,
        ),
      ),
      child: Row(
        children: [
          // ── Toggle speaker icon ──────────────────────
          GestureDetector(
            onTap: _toggle,
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 200),
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: enabled
                    ? AppColors.primary.withOpacity(.15)
                    : AppColors.surface,
                borderRadius: BorderRadius.circular(10),
              ),
              child: Icon(
                enabled
                    ? (vol < 0.01
                        ? Icons.volume_mute_rounded
                        : vol < 0.5
                            ? Icons.volume_down_rounded
                            : Icons.volume_up_rounded)
                    : Icons.volume_off_rounded,
                color: enabled ? AppColors.primary : AppColors.textTertiary,
                size: 20,
              ),
            ),
          ),

          const SizedBox(width: 12),

          // ── Label ───────────────────────────────────
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                'Âm thanh cảnh báo',
                style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                      color: enabled
                          ? AppColors.textPrimary
                          : AppColors.textTertiary,
                      fontWeight: FontWeight.w500,
                      fontSize: 13,
                    ),
              ),
              Text(
                enabled
                    ? '${(vol * 100).round()}%'
                    : 'Đã tắt',
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      fontSize: 11,
                      color: enabled
                          ? AppColors.primary
                          : AppColors.textTertiary,
                    ),
              ),
            ],
          ),

          const SizedBox(width: 8),

          // ── Volume slider ────────────────────────────
          Expanded(
            child: SliderTheme(
              data: SliderTheme.of(context).copyWith(
                activeTrackColor:
                    enabled ? AppColors.primary : AppColors.divider,
                inactiveTrackColor: AppColors.surfaceAlt,
                thumbColor:
                    enabled ? AppColors.primary : AppColors.textTertiary,
                overlayColor: AppColors.primary.withOpacity(.12),
                trackHeight: 3,
                thumbShape:
                    const RoundSliderThumbShape(enabledThumbRadius: 7),
                overlayShape:
                    const RoundSliderOverlayShape(overlayRadius: 16),
              ),
              child: Slider(
                value: vol,
                min: 0,
                max: 1,
                divisions: 20,
                onChanged: enabled ? _onVolumeChange : null,
                onChangeEnd: enabled ? _onVolumeEnd : null,
              ),
            ),
          ),

          // ── Test button ──────────────────────────────
          Tooltip(
            message: 'Thử âm thanh cảnh báo',
            child: AnimatedBuilder(
              animation: _pulseCtrl,
              builder: (_, child) => Transform.scale(
                scale: 1.0 + _pulseCtrl.value * 0.15,
                child: child,
              ),
              child: IconButton(
                onPressed: enabled ? _testSound : null,
                icon: Icon(
                  Icons.notifications_active_rounded,
                  color: enabled ? AppColors.warning : AppColors.textTertiary,
                  size: 20,
                ),
                style: IconButton.styleFrom(
                  backgroundColor: enabled
                      ? AppColors.warning.withOpacity(.12)
                      : Colors.transparent,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    ).animate().fadeIn(duration: 300.ms);
  }
}
