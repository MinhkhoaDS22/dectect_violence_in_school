// lib/screens/monitor_screen.dart
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:file_picker/file_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../theme/app_theme.dart';
import '../models/alert_model.dart';
import '../services/api_service.dart';
import '../services/sound_service.dart';
import '../services/pdf_service.dart';
import '../widgets/sound_control_bar.dart';
import '../widgets/violence_timeline.dart';

enum AnalysisState { idle, uploading, analyzing, done, error }

class MonitorScreen extends StatefulWidget {
  const MonitorScreen({super.key});

  @override
  State<MonitorScreen> createState() => _MonitorScreenState();
}

class _MonitorScreenState extends State<MonitorScreen> {
  final _api = ApiService();

  AnalysisState _state = AnalysisState.idle;
  double _uploadProgress = 0;
  AnalyzeResult? _result;
  String? _errorMsg;
  String? _pickedFileName;

  // Loaded from prefs
  String _email = '';
  String _phone = '';
  String _telegramChatId = '';

  @override
  void initState() {
    super.initState();
    _loadPrefs();
  }

  Future<void> _loadPrefs() async {
    final p = await SharedPreferences.getInstance();
    setState(() {
      _email = p.getString('email') ?? '';
      _phone = p.getString('phone') ?? '';
      _telegramChatId = p.getString('telegram_chat_id') ?? '';
    });
  }

  Future<void> _pickAndAnalyze() async {
    final picked = await FilePicker.platform.pickFiles(
      type: FileType.video,
      withData: true,
    );
    if (picked == null || picked.files.isEmpty) return;

    final file = picked.files.first;
    final bytes = file.bytes;
    if (bytes == null) return;

    setState(() {
      _state = AnalysisState.uploading;
      _uploadProgress = 0;
      _result = null;
      _errorMsg = null;
      _pickedFileName = file.name;
    });

    try {
      final result = await _api.analyzeVideo(
        fileName: file.name,
        fileBytes: bytes,
        email: _email.isEmpty ? null : _email,
        phone: _phone.isEmpty ? null : _phone,
        telegramChatId: _telegramChatId.isEmpty ? null : _telegramChatId,
        onProgress: (p) {
          setState(() {
            _uploadProgress = p;
            if (p >= 1.0) _state = AnalysisState.analyzing;
          });
        },
      );
      setState(() {
        _result = result;
        _state = AnalysisState.done;
      });
      // Phát âm thanh cảnh báo nếu phát hiện bạo lực
      if (result.isViolence) {
        SoundService.instance.playAlert();
      }
    } catch (e) {
      setState(() {
        _state = AnalysisState.error;
        _errorMsg = e.toString();
      });
    }
  }

  void _reset() => setState(() {
        _state = AnalysisState.idle;
        _result = null;
        _errorMsg = null;
        _pickedFileName = null;
        _uploadProgress = 0;
      });

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 720),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Header
              _SectionHeader(
                icon: Icons.video_library_rounded,
                title: 'Phân tích Video',
                subtitle: 'Upload video để phát hiện bạo lực học đường',
              ),
              const SizedBox(height: 24),

              // Sound control bar
              const SoundControlBar(),
              const SizedBox(height: 16),

              // Contact info chip row
              _ContactChips(
                  email: _email,
                  phone: _phone,
                  telegram: _telegramChatId),
              const SizedBox(height: 24),

              // Main card
              AnimatedSwitcher(
                duration: const Duration(milliseconds: 400),
                switchInCurve: Curves.easeOut,
                switchOutCurve: Curves.easeIn,
                child: switch (_state) {
                  AnalysisState.idle => _UploadZone(
                      key: const ValueKey('upload'),
                      onPick: _pickAndAnalyze,
                    ),
                  AnalysisState.uploading => _ProgressCard(
                      key: const ValueKey('uploading'),
                      title: 'Đang tải lên...',
                      subtitle: _pickedFileName ?? '',
                      progress: _uploadProgress,
                      icon: Icons.cloud_upload_rounded,
                      color: AppColors.primary,
                    ),
                  AnalysisState.analyzing => _ProgressCard(
                      key: const ValueKey('analyzing'),
                      title: 'Đang phân tích...',
                      subtitle:
                          'Model CNN-BiLSTM-Attention đang xử lý video',
                      progress: null,
                      icon: Icons.psychology_rounded,
                      color: AppColors.warning,
                    ),
                  AnalysisState.done => _ResultCard(
                      key: const ValueKey('done'),
                      result: _result!,
                      onReset: _reset,
                      api: _api,
                    ),
                  AnalysisState.error => _ErrorCard(
                      key: const ValueKey('error'),
                      message: _errorMsg ?? 'Lỗi không xác định',
                      onRetry: _reset,
                    ),
                },
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ── Upload zone ───────────────────────────────────────────
class _UploadZone extends StatefulWidget {
  final VoidCallback onPick;
  const _UploadZone({super.key, required this.onPick});

  @override
  State<_UploadZone> createState() => _UploadZoneState();
}

class _UploadZoneState extends State<_UploadZone> {
  bool _hovering = false;

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      onEnter: (_) => setState(() => _hovering = true),
      onExit: (_) => setState(() => _hovering = false),
      cursor: SystemMouseCursors.click,
      child: GestureDetector(
        onTap: widget.onPick,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          height: 280,
          decoration: BoxDecoration(
            color: _hovering
                ? AppColors.primary.withOpacity(.06)
                : AppColors.surface,
            borderRadius: BorderRadius.circular(20),
            border: Border.all(
              color: _hovering ? AppColors.primary : AppColors.divider,
              width: _hovering ? 2 : 1,
              strokeAlign: BorderSide.strokeAlignInside,
            ),
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                width: 80,
                height: 80,
                decoration: BoxDecoration(
                  color: AppColors.primary.withOpacity(.1),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Icon(Icons.video_file_rounded,
                    size: 40,
                    color: _hovering
                        ? AppColors.primary
                        : AppColors.textSecondary),
              ).animate(target: _hovering ? 1 : 0).scale(
                    begin: const Offset(1, 1),
                    end: const Offset(1.08, 1.08),
                  ),
              const SizedBox(height: 20),
              Text('Chọn hoặc kéo thả video',
                  style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 8),
              Text(
                'Hỗ trợ: MP4, AVI, MOV, MKV',
                style: Theme.of(context).textTheme.bodyMedium,
              ),
              const SizedBox(height: 24),
              OutlinedButton.icon(
                onPressed: widget.onPick,
                icon: const Icon(Icons.folder_open_rounded, size: 18),
                label: const Text('Chọn video'),
              ),
            ],
          ),
        ),
      ),
    ).animate().fadeIn(duration: 300.ms);
  }
}

// ── Progress card ─────────────────────────────────────────
class _ProgressCard extends StatelessWidget {
  final String title, subtitle;
  final double? progress;
  final IconData icon;
  final Color color;

  const _ProgressCard({
    super.key,
    required this.title,
    required this.subtitle,
    required this.progress,
    required this.icon,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(40),
        child: Column(
          children: [
            Container(
              width: 72,
              height: 72,
              decoration: BoxDecoration(
                color: color.withOpacity(.12),
                shape: BoxShape.circle,
              ),
              child: Icon(icon, size: 36, color: color),
            )
                .animate(onPlay: (c) => c.repeat())
                .shimmer(
                    duration: 1200.ms,
                    color: color.withOpacity(.3)),
            const SizedBox(height: 24),
            Text(title, style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 8),
            Text(subtitle,
                style: Theme.of(context).textTheme.bodyLarge,
                textAlign: TextAlign.center),
            const SizedBox(height: 32),
            if (progress != null)
              Column(
                children: [
                  ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(
                      value: progress,
                      minHeight: 6,
                      backgroundColor: AppColors.surfaceAlt,
                      valueColor:
                          AlwaysStoppedAnimation<Color>(color),
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    '${(progress! * 100).toStringAsFixed(0)}%',
                    style: Theme.of(context)
                        .textTheme
                        .bodyMedium
                        ?.copyWith(color: color),
                  ),
                ],
              )
            else
              const CircularProgressIndicator(
                color: AppColors.warning,
                strokeWidth: 3,
              ),
          ],
        ),
      ),
    ).animate().fadeIn(duration: 300.ms);
  }
}

// ── Result card ───────────────────────────────────────────
class _ResultCard extends StatelessWidget {
  final AnalyzeResult result;
  final VoidCallback onReset;
  final ApiService api;

  const _ResultCard({
    super.key,
    required this.result,
    required this.onReset,
    required this.api,
  });

  @override
  Widget build(BuildContext context) {
    final isVio = result.isViolence;
    final color = isVio ? AppColors.danger : AppColors.success;
    final bgColor = isVio ? AppColors.dangerSoft : AppColors.successSoft;

    return Column(
      children: [
        // Result banner
        Container(
          padding: const EdgeInsets.all(24),
          decoration: BoxDecoration(
            color: bgColor,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: color.withOpacity(.3)),
          ),
          child: Column(
            children: [
              Icon(
                isVio
                    ? Icons.warning_amber_rounded
                    : Icons.check_circle_rounded,
                size: 56,
                color: color,
              ).animate().scale(begin: const Offset(.5, .5)).shake(
                    hz: isVio ? 3 : 0,
                    duration: 600.ms,
                  ),
              const SizedBox(height: 16),
              Text(
                isVio ? '🚨 PHÁT HIỆN BẠO LỰC' : '✅ VIDEO BÌNH THƯỜNG',
                style: Theme.of(context).textTheme.displayMedium?.copyWith(
                      color: color,
                      fontSize: 20,
                    ),
              ),
              const SizedBox(height: 8),
              Text(
                result.summary,
                textAlign: TextAlign.center,
                style: Theme.of(context)
                    .textTheme
                    .bodyLarge
                    ?.copyWith(color: color.withOpacity(.8)),
              ),
            ],
          ),
        ).animate().fadeIn(duration: 400.ms).slideY(begin: .2),

        const SizedBox(height: 16),

        // Stats row
        _StatsRow(result: result),

        // Violence segments
        if (isVio && result.segments.isNotEmpty) ...[
          const SizedBox(height: 16),
          _SegmentsCard(result: result, api: api),
        ],

        // Timeline Visualizer
        if (result.videoDuration > 0) ...[
          const SizedBox(height: 16),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: ViolenceTimeline(
                videoDuration: result.videoDuration,
                segments: result.segments,
              ),
            ),
          ).animate().fadeIn(delay: 250.ms),
        ],

        // Notification status
        const SizedBox(height: 16),
        _NotifStatusCard(status: result.notificationStatus),

        // Actions
        const SizedBox(height: 24),
        Row(
          children: [
            Expanded(
              child: OutlinedButton.icon(
                onPressed: onReset,
                icon: const Icon(Icons.refresh_rounded, size: 18),
                label: const Text('Video khác'),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: ElevatedButton.icon(
                onPressed: () => PdfService.exportAnalysisResult(
                  context: context,
                  result: result,
                  fileName: result.jobId,
                ),
                icon: const Icon(Icons.picture_as_pdf_rounded, size: 18),
                label: const Text('Xuất PDF'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFFFF9F0A),
                ),
              ),
            ),
          ],
        ),
      ],
    );
  }
}

class _StatsRow extends StatelessWidget {
  final AnalyzeResult result;
  const _StatsRow({required this.result});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        _StatChip(
          label: 'Thời lượng',
          value: '${result.videoDuration.toStringAsFixed(1)}s',
          icon: Icons.timer_outlined,
        ),
        const SizedBox(width: 8),
        _StatChip(
          label: 'Tỷ lệ BL',
          value: '${(result.violenceRatio * 100).toStringAsFixed(1)}%',
          icon: Icons.bar_chart_rounded,
          color: result.isViolence ? AppColors.danger : AppColors.success,
        ),
        const SizedBox(width: 8),
        _StatChip(
          label: 'Số người BL',
          value: '${result.maxViolentPersons}',
          icon: Icons.groups_rounded,
          color: result.maxViolentPersons > 0 ? AppColors.danger : AppColors.success,
        ),
        const SizedBox(width: 8),
        _StatChip(
          label: 'Clip đã cắt',
          value: '${result.clips.length}',
          icon: Icons.movie_filter_rounded,
          color: AppColors.primary,
        ),
      ],
    );
  }
}

class _StatChip extends StatelessWidget {
  final String label, value;
  final IconData icon;
  final Color color;

  const _StatChip({
    required this.label,
    required this.value,
    required this.icon,
    this.color = AppColors.textSecondary,
  });

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Card(
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 12),
          child: Column(
            children: [
              Icon(icon, color: color, size: 22),
              const SizedBox(height: 8),
              Text(value,
                  style: Theme.of(context)
                      .textTheme
                      .titleLarge
                      ?.copyWith(color: color)),
              Text(label,
                  style: Theme.of(context).textTheme.bodyMedium,
                  textAlign: TextAlign.center),
            ],
          ),
        ),
      ),
    );
  }
}

class _SegmentsCard extends StatelessWidget {
  final AnalyzeResult result;
  final ApiService api;
  const _SegmentsCard({required this.result, required this.api});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.content_cut_rounded,
                    color: AppColors.danger, size: 18),
                const SizedBox(width: 8),
                Text('Đoạn bạo lực đã cắt',
                    style: Theme.of(context)
                        .textTheme
                        .titleMedium
                        ?.copyWith(color: AppColors.danger)),
              ],
            ),
            const SizedBox(height: 16),
            ...result.segments.asMap().entries.map((e) {
              final i = e.key;
              final seg = e.value;
              final clip =
                  i < result.clips.length ? result.clips[i] : null;
              return _SegmentRow(
                index: i + 1,
                segment: seg,
                clipUrl: clip != null
                    ? api.clipUrlSync(result.jobId, clip)
                    : null,
              );
            }),
          ],
        ),
      ),
    ).animate().fadeIn(delay: 200.ms);
  }
}

class _SegmentRow extends StatelessWidget {
  final int index;
  final ViolenceSegment segment;
  final String? clipUrl;

  const _SegmentRow({
    required this.index,
    required this.segment,
    this.clipUrl,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: AppColors.dangerSoft,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: AppColors.danger.withOpacity(.2)),
      ),
      child: Row(
        children: [
          Container(
            width: 32,
            height: 32,
            decoration: BoxDecoration(
              color: AppColors.danger.withOpacity(.2),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Center(
              child: Text('$index',
                  style: const TextStyle(
                      color: AppColors.danger,
                      fontWeight: FontWeight.w700)),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(segment.timeRange,
                    style: Theme.of(context).textTheme.titleMedium),
                Text(
                  'Độ tin cậy: ${segment.confidence.toStringAsFixed(1)}%  •  Dài: ${segment.duration}',
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
              ],
            ),
          ),
          if (clipUrl != null)
            Tooltip(
              message: 'Download clip',
              child: IconButton(
                onPressed: () {
                  // Open clip URL in new tab
                  // ignore: avoid_web_libraries_in_flutter
                  // dart:html not used; open via url_launcher equivalent
                },
                icon: const Icon(Icons.download_rounded,
                    color: AppColors.primary, size: 20),
              ),
            ),
        ],
      ),
    );
  }
}

class _NotifStatusCard extends StatelessWidget {
  final NotificationStatus status;
  const _NotifStatusCard({required this.status});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Trạng thái thông báo',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            if (status.email != null)
              _NotifRow(
                icon: Icons.email_rounded,
                label: 'Gmail',
                status: status.email!,
              ),
            if (status.telegram != null) ...[
              const SizedBox(height: 8),
              _NotifRow(
                icon: Icons.telegram,
                label: 'Telegram',
                status: status.telegram!,
              ),
            ],
            if (status.email == null && status.telegram == null)
              Text('Không phát hiện bạo lực — không gửi thông báo.',
                  style: Theme.of(context).textTheme.bodyMedium),
          ],
        ),
      ),
    );
  }
}

class _NotifRow extends StatelessWidget {
  final IconData icon;
  final String label, status;
  const _NotifRow(
      {required this.icon, required this.label, required this.status});

  @override
  Widget build(BuildContext context) {
    final ok = status == 'sent';
    return Row(
      children: [
        Icon(icon,
            size: 18,
            color: ok ? AppColors.success : AppColors.danger),
        const SizedBox(width: 8),
        Text(label, style: Theme.of(context).textTheme.bodyLarge),
        const Spacer(),
        Container(
          padding:
              const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
          decoration: BoxDecoration(
            color: ok
                ? AppColors.successSoft
                : AppColors.dangerSoft,
            borderRadius: BorderRadius.circular(20),
          ),
          child: Text(
            ok ? 'Đã gửi ✓' : 'Thất bại ✗',
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: ok ? AppColors.success : AppColors.danger,
            ),
          ),
        ),
      ],
    );
  }
}

// ── Error card ────────────────────────────────────────────
class _ErrorCard extends StatelessWidget {
  final String message;
  final VoidCallback onRetry;
  const _ErrorCard({super.key, required this.message, required this.onRetry});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(40),
        child: Column(
          children: [
            const Icon(Icons.error_outline_rounded,
                size: 56, color: AppColors.danger),
            const SizedBox(height: 16),
            Text('Đã xảy ra lỗi',
                style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 8),
            Text(message,
                textAlign: TextAlign.center,
                style: Theme.of(context)
                    .textTheme
                    .bodyLarge
                    ?.copyWith(color: AppColors.danger, fontSize: 12)),
            const SizedBox(height: 24),
            ElevatedButton.icon(
              onPressed: onRetry,
              icon: const Icon(Icons.refresh_rounded),
              label: const Text('Thử lại'),
              style: ElevatedButton.styleFrom(
                  backgroundColor: AppColors.danger),
            ),
          ],
        ),
      ),
    ).animate().fadeIn(duration: 300.ms);
  }
}

// ── Helpers ───────────────────────────────────────────────
class _SectionHeader extends StatelessWidget {
  final IconData icon;
  final String title, subtitle;
  const _SectionHeader(
      {required this.icon, required this.title, required this.subtitle});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Container(
          width: 48,
          height: 48,
          decoration: BoxDecoration(
            color: AppColors.primary.withOpacity(.12),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Icon(icon, color: AppColors.primary, size: 24),
        ),
        const SizedBox(width: 16),
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title,
                style: Theme.of(context).textTheme.displayMedium),
            Text(subtitle,
                style: Theme.of(context).textTheme.bodyLarge),
          ],
        ),
      ],
    );
  }
}

class _ContactChips extends StatelessWidget {
  final String email, phone, telegram;
  const _ContactChips(
      {required this.email, required this.phone, required this.telegram});

  @override
  Widget build(BuildContext context) {
    final items = [
      if (email.isNotEmpty) (Icons.email_rounded, email),
      if (phone.isNotEmpty) (Icons.phone_rounded, phone),
      if (telegram.isNotEmpty) (Icons.telegram, 'Chat: $telegram'),
    ];
    if (items.isEmpty) return const SizedBox.shrink();
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: items.map((item) {
        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          decoration: BoxDecoration(
            color: AppColors.surfaceAlt,
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: AppColors.divider),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(item.$1, size: 14, color: AppColors.primary),
              const SizedBox(width: 6),
              Text(item.$2,
                  style: Theme.of(context)
                      .textTheme
                      .bodyMedium
                      ?.copyWith(fontSize: 12)),
            ],
          ),
        );
      }).toList(),
    );
  }
}
