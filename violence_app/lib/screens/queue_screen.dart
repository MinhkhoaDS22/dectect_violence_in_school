// lib/screens/queue_screen.dart
//
// QueueScreen — Màn hình hàng chờ phân tích video (multi-camera support).
// Giải pháp câu hỏi "nhiều camera + hàng chờ":
//   • Cho phép chọn nhiều video cùng lúc → thêm vào queue.
//   • QueueService xử lý tuần tự tự động (không cần Redis/Celery).
//   • Mỗi job hiển thị tiến trình real-time: chờ → upload → phân tích → kết quả.
//   • Kết quả mở rộng khi click để xem chi tiết + Timeline.

import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:file_picker/file_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../theme/app_theme.dart';
import '../models/queue_job.dart';
import '../services/queue_service.dart';
import '../services/api_service.dart';
import '../services/pdf_service.dart';
import '../widgets/violence_timeline.dart';

class QueueScreen extends StatefulWidget {
  const QueueScreen({super.key});

  @override
  State<QueueScreen> createState() => _QueueScreenState();
}

class _QueueScreenState extends State<QueueScreen> {
  String _email = '';
  String _phone = '';
  String _telegram = '';

  @override
  void initState() {
    super.initState();
    _loadPrefs();
    QueueService.instance.addListener(_rebuild);
  }

  @override
  void dispose() {
    QueueService.instance.removeListener(_rebuild);
    super.dispose();
  }

  void _rebuild() {
    if (mounted) setState(() {});
  }

  Future<void> _loadPrefs() async {
    final p = await SharedPreferences.getInstance();
    if (mounted) {
      setState(() {
        _email    = p.getString('email') ?? '';
        _phone    = p.getString('phone') ?? '';
        _telegram = p.getString('telegram_chat_id') ?? '';
      });
    }
  }

  Future<void> _addVideos() async {
    final picked = await FilePicker.platform.pickFiles(
      type: FileType.video,
      withData: true,
      allowMultiple: true,
    );
    if (picked == null) return;

    for (final file in picked.files) {
      if (file.bytes == null) continue;
      final id = '${DateTime.now().microsecondsSinceEpoch}_${file.name}';
      final job = QueueJob(
        id: id,
        fileName: file.name,
        bytes: file.bytes!,
      );
      attachContactInfo(
        id,
        email:    _email.isEmpty    ? null : _email,
        phone:    _phone.isEmpty    ? null : _phone,
        telegram: _telegram.isEmpty ? null : _telegram,
      );
      QueueService.instance.addJob(job);
    }
  }

  @override
  Widget build(BuildContext context) {
    final qs   = QueueService.instance;
    final jobs = qs.jobs;

    return Padding(
      padding: const EdgeInsets.all(24),
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 720),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // ── Header ─────────────────────────────────────
              Row(
                children: [
                  Container(
                    width: 48, height: 48,
                    decoration: BoxDecoration(
                      color: AppColors.primary.withOpacity(.12),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Icon(Icons.queue_rounded,
                        color: AppColors.primary, size: 24),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Hàng chờ phân tích',
                            style: Theme.of(context).textTheme.displayMedium),
                        Text(
                          '${qs.waiting} chờ · ${qs.running} đang chạy · '
                          '${qs.completed} hoàn thành',
                          style: Theme.of(context).textTheme.bodyLarge,
                        ),
                      ],
                    ),
                  ),
                  if (jobs.any((j) => j.isFinished))
                    TextButton.icon(
                      onPressed: qs.clearFinished,
                      icon: const Icon(Icons.clear_all_rounded, size: 18),
                      label: const Text('Xoá xong'),
                      style: TextButton.styleFrom(
                          foregroundColor: AppColors.textSecondary),
                    ),
                ],
              ),
              const SizedBox(height: 16),

              // ── Stats ──────────────────────────────────────
              if (jobs.isNotEmpty) ...[
                _StatsRow(qs: qs),
                const SizedBox(height: 16),
              ],

              // ── Add button ─────────────────────────────────
              OutlinedButton.icon(
                onPressed: _addVideos,
                icon: const Icon(Icons.add_rounded),
                label: const Text('Thêm video (nhiều camera)'),
                style: OutlinedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 14),
                ),
              ),
              const SizedBox(height: 4),
              Text(
                'Mỗi video sẽ được xử lý tuần tự tự động — không cần chờ thủ công.',
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: AppColors.textTertiary,
                      fontSize: 11,
                    ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16),

              // ── Job list ───────────────────────────────────
              Expanded(
                child: jobs.isEmpty
                    ? _EmptyQueue(onAdd: _addVideos)
                    : ListView.separated(
                        itemCount: jobs.length,
                        separatorBuilder: (_, __) =>
                            const SizedBox(height: 10),
                        itemBuilder: (_, i) => _JobCard(
                          job: jobs[i],
                          index: i,
                          onRemove: () =>
                              QueueService.instance.removeJob(jobs[i].id),
                        ),
                      ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ── Stats ─────────────────────────────────────────────────
class _StatsRow extends StatelessWidget {
  final QueueService qs;
  const _StatsRow({required this.qs});

  @override
  Widget build(BuildContext context) => Row(
        children: [
          _StatBox(label: 'Tổng',       value: '${qs.jobs.length}', color: AppColors.primary),
          const SizedBox(width: 8),
          _StatBox(label: 'Đang chạy', value: '${qs.running}',  color: AppColors.warning),
          const SizedBox(width: 8),
          _StatBox(label: 'BL',         value: '${qs.violenceCount}', color: AppColors.danger),
          const SizedBox(width: 8),
          _StatBox(label: 'Xong',       value: '${qs.completed}', color: AppColors.success),
        ],
      );
}

class _StatBox extends StatelessWidget {
  final String label, value;
  final Color color;
  const _StatBox({required this.label, required this.value, required this.color});

  @override
  Widget build(BuildContext context) => Expanded(
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 10),
          decoration: BoxDecoration(
            color: color.withOpacity(.08),
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: color.withOpacity(.2)),
          ),
          child: Column(children: [
            Text(value,
                style: Theme.of(context)
                    .textTheme
                    .titleLarge
                    ?.copyWith(color: color, fontWeight: FontWeight.w700)),
            Text(label,
                style: Theme.of(context)
                    .textTheme
                    .bodyMedium
                    ?.copyWith(fontSize: 10),
                textAlign: TextAlign.center),
          ]),
        ),
      );
}

// ── Job Card ──────────────────────────────────────────────
class _JobCard extends StatelessWidget {
  final QueueJob job;
  final int index;
  final VoidCallback onRemove;
  const _JobCard({required this.job, required this.index, required this.onRemove});

  @override
  Widget build(BuildContext context) {
    final analyzeResult = job.result as AnalyzeResult?;
    final canExpand = job.status == JobStatus.done && analyzeResult != null;

    return Card(
      child: canExpand
          ? ExpansionTile(
              leading: _StatusIcon(job: job),
              title: _JobTitle(job: job),
              subtitle: _JobSubtitle(job: job),
              trailing: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  IconButton(
                    icon: const Icon(Icons.picture_as_pdf_rounded,
                        size: 18, color: AppColors.warning),
                    onPressed: () => PdfService.exportAnalysisResult(
                      context: context,
                      result: analyzeResult,
                      fileName: job.fileName,
                    ),
                    tooltip: 'Xuất PDF',
                  ),
                  IconButton(
                    icon: const Icon(Icons.close_rounded,
                        size: 18, color: AppColors.textTertiary),
                    onPressed: onRemove,
                    tooltip: 'Xoá',
                  ),
                ],
              ),
              children: [
                Padding(
                  padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                  child: _JobResult(result: analyzeResult),
                ),
              ],
            )
          : ListTile(
              leading: _StatusIcon(job: job),
              title: _JobTitle(job: job),
              subtitle: _JobSubtitle(job: job),
              trailing: job.isFinished
                  ? IconButton(
                      icon: const Icon(Icons.close_rounded,
                          size: 18, color: AppColors.textTertiary),
                      onPressed: onRemove,
                    )
                  : null,
            ),
    )
        .animate()
        .fadeIn(delay: (index * 50).ms, duration: 300.ms)
        .slideX(begin: .05);
  }
}

class _StatusIcon extends StatelessWidget {
  final QueueJob job;
  const _StatusIcon({required this.job});

  @override
  Widget build(BuildContext context) {
    final w = Container(
      width: 44, height: 44,
      decoration: BoxDecoration(
        color: job.color.withOpacity(.12),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Icon(job.icon, color: job.color, size: 22),
    );
    if (job.isRunning) {
      return w
          .animate(onPlay: (c) => c.repeat())
          .shimmer(duration: 1200.ms, color: job.color.withOpacity(.4));
    }
    return w;
  }
}

class _JobTitle extends StatelessWidget {
  final QueueJob job;
  const _JobTitle({required this.job});

  @override
  Widget build(BuildContext context) => Text(
        job.fileName,
        style: Theme.of(context).textTheme.titleMedium,
        maxLines: 1,
        overflow: TextOverflow.ellipsis,
      );
}

class _JobSubtitle extends StatelessWidget {
  final QueueJob job;
  const _JobSubtitle({required this.job});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(job.statusLabel,
            style: Theme.of(context)
                .textTheme
                .bodyMedium
                ?.copyWith(color: job.color, fontSize: 12)),
        if (job.status == JobStatus.uploading)
          Padding(
            padding: const EdgeInsets.only(top: 4),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(3),
              child: LinearProgressIndicator(
                value: job.progress,
                minHeight: 3,
                backgroundColor: AppColors.surfaceAlt,
                valueColor: AlwaysStoppedAnimation<Color>(job.color),
              ),
            ),
          ),
        if (job.status == JobStatus.analyzing)
          Padding(
            padding: const EdgeInsets.only(top: 4),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(3),
              child: const LinearProgressIndicator(
                minHeight: 3,
                backgroundColor: AppColors.surfaceAlt,
              ),
            ),
          ),
      ],
    );
  }
}

class _JobResult extends StatelessWidget {
  final AnalyzeResult result;
  const _JobResult({required this.result});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Divider(),
        const SizedBox(height: 8),
        Text(result.summary,
            style: Theme.of(context).textTheme.bodyMedium),
        const SizedBox(height: 12),
        if (result.videoDuration > 0)
          ViolenceTimeline(
            videoDuration: result.videoDuration,
            segments: result.segments,
          ),
      ],
    );
  }
}

// ── Empty state ───────────────────────────────────────────
class _EmptyQueue extends StatelessWidget {
  final VoidCallback onAdd;
  const _EmptyQueue({required this.onAdd});

  @override
  Widget build(BuildContext context) => Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.video_library_outlined,
                size: 64, color: AppColors.textTertiary),
            const SizedBox(height: 16),
            Text('Hàng chờ trống',
                style: Theme.of(context)
                    .textTheme
                    .titleMedium
                    ?.copyWith(color: AppColors.textTertiary)),
            const SizedBox(height: 8),
            Text(
              'Thêm video từ nhiều camera.\nHệ thống sẽ phân tích tuần tự tự động.',
              style: Theme.of(context).textTheme.bodyLarge,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 24),
            ElevatedButton.icon(
              onPressed: onAdd,
              icon: const Icon(Icons.add_rounded),
              label: const Text('Thêm video'),
            ),
          ],
        ),
      ).animate().fadeIn();
}
